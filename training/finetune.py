"""
Stage 2: 설문 데이터로 앙상블 모델 파인튜닝.

Stage 1에서 학습된 오디오 백본 가중치를 로드하고,
텍스트 모델과 결합하여 실제 설문 점수(1~5)를 예측하도록 훈련한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

from config import train_cfg, data_cfg, audio_cfg
from data.prepare_dataset import load_survey_dataset, split_dataset, build_dataloaders
from data.augmentation import AudioAugmentor
from models.ensemble import SsagajiEnsemble
from training.train_utils import (
    set_seed, get_device, build_optimizer, build_scheduler,
    RMSELoss, EarlyStopper, save_checkpoint, load_checkpoint
)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    rmse = torch.sqrt(((preds - targets) ** 2).mean()).item()
    mae = (preds - targets).abs().mean().item()
    # Pearson r
    vx = preds - preds.mean()
    vy = targets - targets.mean()
    pearson = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)
    return {"rmse": rmse, "mae": mae, "pearson": pearson.item()}


def finetune(
    survey_csv: str,
    stage1_ckpt: str = None,
    output_dir: str = "checkpoints/best_model",
    fusion: str = "concat",
):
    set_seed()
    device = get_device()
    print(f"[Stage 2] Device: {device}")

    # 데이터 준비
    df = load_survey_dataset(Path(survey_csv))
    augmentor = AudioAugmentor(seed=train_cfg.seed)
    df = augmentor.augment_dataset(df, data_cfg.data_dir, n_variants=data_cfg.augmentation_factor)
    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        batch_size=train_cfg.stage2_batch_size,
    )

    # 앙상블 모델
    model = SsagajiEnsemble(fusion=fusion).to(device)

    # Stage 1 가중치 로드
    if stage1_ckpt:
        epoch, _ = load_checkpoint(model.audio_model, Path(stage1_ckpt), device)
        print(f"  Stage 1 체크포인트 로드 완료 (epoch {epoch})")
        # 회귀 헤드로 전환 (감정 분류 헤드 → 회귀 헤드)
        model.audio_model.set_stage("regression")

    optimizer = build_optimizer(model, lr=train_cfg.stage2_lr)
    total_steps = len(train_loader) * train_cfg.stage2_epochs
    scheduler = build_scheduler(optimizer, total_steps)
    criterion = RMSELoss()
    stopper = EarlyStopper(patience=train_cfg.patience)

    best_val_rmse = float("inf")
    for epoch in range(1, train_cfg.stage2_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.stage2_epochs}"):
            input_values = batch["input_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(input_values, input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # 검증
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                preds = model(input_values, input_ids, attention_mask)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)

        print(
            f"  Epoch {epoch}: train_rmse={avg_train:.4f} | "
            f"val_rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}, r={metrics['pearson']:.4f}"
        )

        if metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            save_checkpoint(model, optimizer, epoch, metrics["rmse"], Path(output_dir))
            # 모델 전체 저장
            torch.save(model.state_dict(), Path(output_dir) / "model_weights.pt")
            print(f"  → 최적 모델 저장 (val_rmse={metrics['rmse']:.4f})")

        if stopper(metrics["rmse"]):
            print("[Stage 2] 조기 종료")
            break

    # 테스트 평가
    print("\n[최종 테스트 평가]")
    load_checkpoint(model, Path(output_dir), device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_values, input_ids, attention_mask)
            all_preds.append(preds.cpu())
            all_labels.append(batch["label"])

    test_metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
    print(f"  RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, Pearson r={test_metrics['pearson']:.4f}")
    return test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_csv", required=True, help="설문 결과 CSV 경로")
    parser.add_argument("--stage1_ckpt", default=None, help="Stage 1 체크포인트 디렉터리")
    parser.add_argument("--output_dir", default="checkpoints/best_model")
    parser.add_argument("--fusion", default="concat", choices=["weighted", "learned", "concat"])
    args = parser.parse_args()
    finetune(args.survey_csv, args.stage1_ckpt, args.output_dir, args.fusion)
