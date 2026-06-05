"""
평가 지표 및 시각화 모듈.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, mean_absolute_error


def compute_all_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    mae = mean_absolute_error(targets, preds)
    pearson_r, pearson_p = pearsonr(preds, targets)
    spearman_r, spearman_p = spearmanr(preds, targets)

    # 반올림 후 정확도 (1~5 정수 클래스)
    pred_int = np.clip(np.round(preds), 1, 5).astype(int)
    tgt_int = np.clip(np.round(targets), 1, 5).astype(int)
    acc = (pred_int == tgt_int).mean()
    within_1 = (np.abs(pred_int - tgt_int) <= 1).mean()

    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "Pearson_r": round(float(pearson_r), 4),
        "Pearson_p": round(float(pearson_p), 4),
        "Spearman_r": round(float(spearman_r), 4),
        "Accuracy@1": round(float(acc), 4),
        "Accuracy@1±1": round(float(within_1), 4),
    }


def plot_score_distribution(preds: np.ndarray, targets: np.ndarray, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 분포 비교
    axes[0].hist(targets, bins=20, alpha=0.6, label="실제 점수", color="steelblue")
    axes[0].hist(preds, bins=20, alpha=0.6, label="예측 점수", color="coral")
    axes[0].set_xlabel("점수 (1~5)")
    axes[0].set_ylabel("빈도")
    axes[0].set_title("점수 분포 비교")
    axes[0].legend()

    # 2. 산점도
    axes[1].scatter(targets, preds, alpha=0.4, s=20)
    axes[1].plot([1, 5], [1, 5], "r--", label="완벽 예측")
    axes[1].set_xlabel("실제 점수")
    axes[1].set_ylabel("예측 점수")
    axes[1].set_title("예측 vs 실제")
    axes[1].legend()

    # 3. 혼동 행렬
    pred_int = np.clip(np.round(preds), 1, 5).astype(int)
    tgt_int = np.clip(np.round(targets), 1, 5).astype(int)
    cm = confusion_matrix(tgt_int, pred_int, labels=[1, 2, 3, 4, 5])
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[2], cmap="Blues",
                xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    axes[2].set_xlabel("예측")
    axes[2].set_ylabel("실제")
    axes[2].set_title("혼동 행렬 (반올림)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[평가] 그래프 저장: {save_path}")
    else:
        plt.show()
    plt.close()


def evaluate_model(model, data_loader, device: torch.device, save_plot: str = None) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_values = batch["input_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            preds = model(input_values, input_ids, attention_mask).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    metrics = compute_all_metrics(preds_np, labels_np)
    print("\n[평가 결과]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if save_plot:
        plot_score_distribution(preds_np, labels_np, save_path=save_plot)

    return metrics


def inter_rater_reliability(survey_csv: str) -> dict:
    """설문 데이터의 평가자 간 신뢰도를 계산한다 (Krippendorff's alpha 근사)."""
    import re
    df = pd.read_csv(survey_csv)
    rater_cols = [c for c in df.columns if re.match(r"rater_\d+", c)]
    ratings = df[rater_cols].values.astype(float)

    # Krippendorff's alpha (ordinal)
    n_items, n_raters = ratings.shape
    grand_mean = np.nanmean(ratings)

    # 관측된 불일치
    do = 0.0
    count = 0
    for i in range(n_items):
        row = ratings[i][~np.isnan(ratings[i])]
        for j in range(len(row)):
            for k in range(j + 1, len(row)):
                do += (row[j] - row[k]) ** 2
                count += 1

    # 기대 불일치
    all_vals = ratings[~np.isnan(ratings)]
    de = np.sum((all_vals - grand_mean) ** 2) * 2 / (len(all_vals) - 1) if len(all_vals) > 1 else 1

    alpha = 1 - (do / count) / de if (count > 0 and de > 0) else 0

    result = {
        "krippendorff_alpha": round(float(alpha), 4),
        "n_raters": n_raters,
        "n_items": n_items,
        "mean_score": round(float(grand_mean), 4),
        "std_score": round(float(np.nanstd(ratings)), 4),
    }
    print(f"[신뢰도] Krippendorff α = {alpha:.4f} (0.8↑: 우수, 0.67↑: 허용)")
    return result
