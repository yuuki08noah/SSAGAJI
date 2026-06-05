"""
감정 사전학습 유무에 따른 성능 비교 실험.

실제 HuBERT 대신 구조가 동일한 경량 1D-CNN 백본을 사용해
CPU에서 수 분 내 실행 가능하도록 설계했다.

핵심 질문:
  감정 분류로 사전학습한 백본이 처음부터 회귀 학습하는 것보다
  더 빠르게 수렴하고 최종 성능이 더 좋은가?

실행:
    python scripts/compare_pretraining.py
    python scripts/compare_pretraining.py --epochs 30 --output results/
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import audio_cfg, data_cfg

# ─────────────────────────────────────────────
# 1. 경량 백본 (HuBERT와 구조적으로 동일한 역할)
# ─────────────────────────────────────────────

class LightAudioBackbone(nn.Module):
    """
    1D-CNN 경량 백본 (HuBERT의 CNN 특징 추출기 역할).
    CPU에서 수 초 내 학습 가능하도록 의도적으로 작게 설계.
    """
    HIDDEN = 64

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=20, stride=10), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=8, stride=4),  nn.GELU(),
            nn.Conv1d(32, self.HIDDEN, kernel_size=4, stride=2), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x.unsqueeze(1))   # (B, H, T')
        return x.mean(dim=-1)           # (B, H) — 평균 풀링


class EmotionHead(nn.Module):
    def __init__(self, hidden: int, num_emotions: int):
        super().__init__()
        self.fc = nn.Linear(hidden, num_emotions)

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.net(x)).squeeze(-1) * 4 + 1   # [1, 5]


# ─────────────────────────────────────────────
# 2. 데이터셋
# ─────────────────────────────────────────────

SR = 16000
MAX_SAMPLES = SR * 3   # 3초

def load_wav(path: str) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    w = torch.from_numpy(data)
    if sr != SR:
        # 간단한 리샘플링 (scipy 없이)
        ratio = SR / sr
        n = int(len(w) * ratio)
        w = torch.nn.functional.interpolate(
            w.unsqueeze(0).unsqueeze(0), size=n, mode="linear", align_corners=False
        ).squeeze()
    if w.shape[0] > MAX_SAMPLES:
        w = w[:MAX_SAMPLES]
    elif w.shape[0] < MAX_SAMPLES:
        w = torch.nn.functional.pad(w, (0, MAX_SAMPLES - w.shape[0]))
    return w


class EmotionDataset(Dataset):
    EMOTIONS = list(data_cfg.emotion_to_rudeness.keys())
    E2I = {e: i for i, e in enumerate(EMOTIONS)}

    def __init__(self, root: Path):
        self.items = []
        for wav in sorted(root.rglob("*.wav")):
            emotion = wav.parent.name
            if emotion in self.E2I:
                self.items.append((str(wav), self.E2I[emotion]))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return load_wav(path), torch.tensor(label, dtype=torch.long)


class SurveyDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        rater_cols = [c for c in df.columns if c.startswith("rater_")]
        df = df.copy()
        df["label"] = df[rater_cols].mean(axis=1)
        self.items = list(zip(df["audio_path"].tolist(), df["label"].tolist()))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return load_wav(path), torch.tensor(label, dtype=torch.float)


# ─────────────────────────────────────────────
# 3. 학습 루프
# ─────────────────────────────────────────────

def rmse(pred, target):
    return torch.sqrt(((pred - target) ** 2).mean())


def train_emotion(backbone, head, loader, optimizer, device):
    backbone.train(); head.train()
    total = 0
    criterion = nn.CrossEntropyLoss()
    for wav, label in loader:
        wav, label = wav.to(device), label.to(device)
        optimizer.zero_grad()
        feat = backbone(wav)
        loss = criterion(head(feat), label)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_regression(backbone, head, loader, optimizer, device):
    backbone.train(); head.train()
    total = 0
    for wav, label in loader:
        wav, label = wav.to(device), label.to(device)
        optimizer.zero_grad()
        feat = backbone(wav)
        loss = rmse(head(feat), label)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval(); head.eval()
    preds, targets = [], []
    for wav, label in loader:
        wav = wav.to(device)
        feat = backbone(wav)
        preds.append(head(feat).cpu())
        targets.append(label)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    r = rmse(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    vx = preds - preds.mean(); vy = targets - targets.mean()
    pearson = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)
    return {"rmse": r, "mae": mae, "pearson": pearson.item()}


# ─────────────────────────────────────────────
# 4. 두 경로 실험
# ─────────────────────────────────────────────

def run_experiment(
    survey_csv: Path,
    aihub_dir: Path,
    epochs: int = 25,
    lr: float = 1e-3,
    seed: int = 42,
    device_str: str = "cpu",
):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device(device_str)

    # ── 데이터 준비 ──
    df = pd.read_csv(survey_csv)
    rater_cols = [c for c in df.columns if c.startswith("rater_")]
    df["label"] = df[rater_cols].mean(axis=1)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_ds = SurveyDataset(train_df.reset_index(drop=True))
    val_ds   = SurveyDataset(val_df.reset_index(drop=True))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8)

    emotion_ds = EmotionDataset(aihub_dir)
    emotion_loader = DataLoader(emotion_ds, batch_size=8, shuffle=True)
    num_emotions = len(EmotionDataset.EMOTIONS)

    H = LightAudioBackbone.HIDDEN

    results = {}

    for with_pretrain in [False, True]:
        label = "사전학습 O" if with_pretrain else "사전학습 X"
        print(f"\n{'='*50}")
        print(f"  실험: {label}")
        print(f"{'='*50}")

        backbone = LightAudioBackbone().to(device)
        reg_head = RegressionHead(H).to(device)

        # ── Stage 1: 감정 사전학습 ──
        if with_pretrain:
            emo_head = EmotionHead(H, num_emotions).to(device)
            opt = torch.optim.Adam(
                list(backbone.parameters()) + list(emo_head.parameters()), lr=lr
            )
            print("  [Stage 1] 감정 분류 사전학습...")
            for ep in range(10):
                loss = train_emotion(backbone, emo_head, emotion_loader, opt, device)
                if (ep + 1) % 5 == 0:
                    print(f"    epoch {ep+1:02d}: emotion_loss={loss:.4f}")
            del emo_head

        # ── Stage 2: 회귀 파인튜닝 ──
        opt = torch.optim.Adam(
            list(backbone.parameters()) + list(reg_head.parameters()), lr=lr * 0.5
        )
        history = {"train_rmse": [], "val_rmse": [], "val_mae": [], "val_pearson": []}

        print(f"  [Stage 2] 회귀 파인튜닝 ({epochs} epochs)...")
        for ep in range(epochs):
            tr = train_regression(backbone, reg_head, train_loader, opt, device)
            val = evaluate(backbone, reg_head, val_loader, device)
            history["train_rmse"].append(tr)
            history["val_rmse"].append(val["rmse"])
            history["val_mae"].append(val["mae"])
            history["val_pearson"].append(val["pearson"])

            if (ep + 1) % 5 == 0:
                print(
                    f"    epoch {ep+1:02d}: "
                    f"train={tr:.4f}  val_rmse={val['rmse']:.4f}  "
                    f"mae={val['mae']:.4f}  r={val['pearson']:.3f}"
                )

        results[label] = history
        final = evaluate(backbone, reg_head, val_loader, device)
        print(f"\n  최종 결과: RMSE={final['rmse']:.4f}  MAE={final['mae']:.4f}  r={final['pearson']:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────

def plot_comparison(results: dict, output_dir: Path, epochs: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("감정 사전학습 유무 비교", fontsize=14, fontweight="bold")

    colors = {"사전학습 O": "#4f46e5", "사전학습 X": "#ef4444"}
    x = list(range(1, epochs + 1))

    # (1) val RMSE 수렴 곡선
    ax = axes[0]
    for label, hist in results.items():
        ax.plot(x, hist["val_rmse"], label=label, color=colors[label], linewidth=2)
    ax.set_title("검증 RMSE (낮을수록 좋음)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) val Pearson r 수렴 곡선
    ax = axes[1]
    for label, hist in results.items():
        ax.plot(x, hist["val_pearson"], label=label, color=colors[label], linewidth=2)
    ax.set_title("검증 Pearson r (높을수록 좋음)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("r")
    ax.legend()
    ax.grid(alpha=0.3)

    # (3) 최종 지표 막대 그래프
    ax = axes[2]
    metrics = ["val_rmse", "val_mae"]
    metric_labels = ["RMSE (↓)", "MAE (↓)"]
    bar_w = 0.3
    n = len(metrics)
    idx = np.arange(n)

    for i, (label, hist) in enumerate(results.items()):
        vals = [hist[m][-1] for m in metrics]
        bars = ax.bar(idx + i * bar_w, vals, bar_w, label=label, color=colors[label], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"최종 성능 비교 (epoch {epochs})")
    ax.set_xticks(idx + bar_w / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[결과 그래프] {out}")

    # 텍스트 요약 출력
    print("\n" + "=" * 50)
    print("  최종 지표 비교 요약")
    print("=" * 50)
    print(f"{'지표':<15} {'사전학습 X':>12} {'사전학습 O':>12} {'개선폭':>10}")
    print("-" * 50)
    for m, lbl in [("val_rmse", "RMSE↓"), ("val_mae", "MAE↓"), ("val_pearson", "Pearson r↑")]:
        no  = results["사전학습 X"][m][-1]
        yes = results["사전학습 O"][m][-1]
        if "pearson" in m:
            diff = f"+{yes - no:.4f}" if yes > no else f"{yes - no:.4f}"
        else:
            diff = f"{no - yes:+.4f}" if no > yes else f"{yes - no:+.4f}"
        print(f"  {lbl:<13} {no:>12.4f} {yes:>12.4f} {diff:>10}")

    # 수렴 속도: val_rmse가 특정 임계값 이하로 처음 도달하는 epoch
    print("\n  수렴 속도 (val_rmse 기준)")
    thresholds = [1.4, 1.3, 1.2]
    for th in thresholds:
        row = []
        for label in ["사전학습 X", "사전학습 O"]:
            hist = results[label]["val_rmse"]
            ep = next((i + 1 for i, v in enumerate(hist) if v <= th), None)
            row.append(f"{ep}epoch" if ep else "미달성")
        print(f"  RMSE≤{th}: 사전학습 X={row[0]}, 사전학습 O={row[1]}")


# ─────────────────────────────────────────────
# 6. 진입점
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_csv", default="data/survey_results.csv")
    parser.add_argument("--aihub_dir",  default="data/aihub_emotion")
    parser.add_argument("--epochs",     type=int, default=25)
    parser.add_argument("--output",     default="results/pretraining_comparison")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    survey_csv = Path(args.survey_csv)
    aihub_dir  = Path(args.aihub_dir)

    # 데이터 없으면 자동 생성
    if not survey_csv.exists() or not aihub_dir.exists():
        print("[데이터 없음] 샘플 데이터를 자동 생성합니다...")
        from scripts.generate_sample_data import (
            generate_survey_audio, generate_survey_csv, generate_aihub_emotion_dummy
        )
        audio_dir = Path("survey_app/static/audio")
        paths = generate_survey_audio(audio_dir)
        generate_survey_csv(paths, survey_csv, n_raters=15)
        generate_aihub_emotion_dummy(aihub_dir, n_per_emotion=30)

    device = "cpu"   # MPS는 소규모 배치에서 CPU보다 느릴 수 있음
    print(f"Device: {device}")

    results = run_experiment(
        survey_csv=survey_csv,
        aihub_dir=aihub_dir,
        epochs=args.epochs,
        seed=args.seed,
        device_str=device,
    )

    plot_comparison(results, Path(args.output), args.epochs)
