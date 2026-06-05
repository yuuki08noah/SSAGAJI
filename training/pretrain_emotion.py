"""
Stage 1: AI Hub 감정 음성 데이터셋으로 오디오 모델 사전학습.

감정 분류 태스크를 통해 분노, 냉소, 조소 등
무례함과 직결되는 음향 특징을 백본에 내재화한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import data_cfg, train_cfg, audio_cfg
from data.prepare_dataset import load_aihub_emotion_dataset, split_dataset, SsagajiDataset
from data.augmentation import AudioAugmentor
from models.audio_model import AudioSsagajiModel
from training.train_utils import (
    set_seed, get_device, build_optimizer, build_scheduler, EarlyStopper, save_checkpoint
)


EMOTION_LABELS = list(data_cfg.emotion_to_rudeness.keys())
NUM_EMOTIONS = len(EMOTION_LABELS)
LABEL2IDX = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_extractor):
        import torchaudio
        self.df = df.reset_index(drop=True)
        self.fe = feature_extractor
        self.max_samples = int(audio_cfg.max_duration * audio_cfg.sample_rate)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torchaudio
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["audio_path"])
        waveform = waveform.mean(0)
        if sr != audio_cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, audio_cfg.sample_rate)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[: self.max_samples]

        inputs = self.fe(
            waveform.numpy(),
            sampling_rate=audio_cfg.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        )
        label_idx = LABEL2IDX.get(row["emotion"], 0)
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


def pretrain(aihub_root: str, output_dir: str = "checkpoints/stage1"):
    set_seed()
    device = get_device()
    print(f"[Stage 1] Device: {device}")

    # 데이터 로드 + 증강
    df = load_aihub_emotion_dataset(Path(aihub_root))
    augmentor = AudioAugmentor(seed=train_cfg.seed)
    df = augmentor.augment_dataset(df, output_dir=data_cfg.data_dir, n_variants=data_cfg.augmentation_factor)

    train_df, val_df, _ = split_dataset(df)

    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(audio_cfg.backbone)

    train_ds = EmotionDataset(train_df, fe)
    val_ds = EmotionDataset(val_df, fe)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.stage1_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.stage1_batch_size, num_workers=4)

    # 모델: 감정 분류 모드
    model = AudioSsagajiModel(num_emotions=NUM_EMOTIONS, stage="emotion").to(device)
    optimizer = build_optimizer(model, lr=train_cfg.stage1_lr)
    total_steps = len(train_loader) * train_cfg.stage1_epochs
    scheduler = build_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=train_cfg.patience)

    best_val_loss = float("inf")
    for epoch in range(1, train_cfg.stage1_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.stage1_epochs}"):
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # 검증
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_values)
                val_loss += criterion(logits, labels).item()
                correct += (logits.argmax(-1) == labels).sum().item()

        val_loss /= len(val_loader)
        acc = correct / len(val_ds)
        print(f"  Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={val_loss:.4f}, val_acc={acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, Path(output_dir))
            print(f"  → 체크포인트 저장 (val_loss={val_loss:.4f})")

        if stopper(val_loss):
            print("[Stage 1] 조기 종료")
            break

    print(f"[Stage 1] 완료. 최적 val_loss: {best_val_loss:.4f}")
    return Path(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aihub_root", required=True, help="AI Hub 감정 데이터셋 루트 경로")
    parser.add_argument("--output_dir", default="checkpoints/stage1")
    args = parser.parse_args()
    pretrain(args.aihub_root, args.output_dir)
