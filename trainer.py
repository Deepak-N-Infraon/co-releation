"""
trainer.py
──────────
Training pipeline with:
  - Train / validation split (time-based, no leakage)
  - Learning rate scheduling
  - Early stopping
  - Live training metrics + plots
  - Post-training analysis (feature importance, confusion matrix, etc.)
  - Model checkpointing
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import ALL_STATS, ALARM_RULES
from model import CorrelationLSTM, CorrelationLoss, ALARM_NAMES, N_STATS, N_ALARMS


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class TrainConfig:
    window_size:    int   = 24
    horizon:        int   = 6
    batch_size:     int   = 128       # smaller than 256 — better gradient signal
    epochs:         int   = 60
    lr:             float = 5e-4      # conservative LR — 3e-3 was too high, caused flat loss
    weight_decay:   float = 1e-4
    patience:       int   = 15        # give it room to improve slowly
    val_split:      float = 0.2
    hidden_dim:     int   = 64
    n_layers:       int   = 1
    n_heads:        int   = 2
    dropout:        float = 0.1       # less dropout — model is small, don't regularise too hard
    w_value:        float = 0.5
    w_breach:       float = 3.0
    w_alarm:        float = 2.0
    max_train_windows: int = 8000
    output_dir:     str   = "outputs"
    device:         str   = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class MetricsDataset(Dataset):
    def __init__(
        self,
        X:        np.ndarray,   # (N, window, features)
        y_values: np.ndarray,   # (N, horizon, 9)
        y_breach: np.ndarray,   # (N, horizon, 9)
        y_alarm:  np.ndarray,   # (N, 13)
    ):
        self.X        = torch.tensor(X,        dtype=torch.float32)
        self.y_values = torch.tensor(y_values, dtype=torch.float32)
        self.y_breach = torch.tensor(y_breach, dtype=torch.float32)
        self.y_alarm  = torch.tensor(y_alarm,  dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_values[idx], self.y_breach[idx], self.y_alarm[idx]


def build_alarm_labels(y_breach: np.ndarray, horizon: int, n_alarms: int = N_ALARMS) -> np.ndarray:
    """
    Collapse per-step breach flags into a single per-window alarm label vector.
    y_breach: (N, horizon, 9)  → y_alarm: (N, 13)
    First 9 alarms map to 9 stats (breach alarms), last 4 are clear alarms.
    """
    N = y_breach.shape[0]
    y_alarm = np.zeros((N, n_alarms), dtype=np.float32)

    # Breach alarms: any breach in horizon → 1
    for stat_idx in range(N_STATS):
        if stat_idx < n_alarms:
            y_alarm[:, stat_idx] = (y_breach[:, :, stat_idx].sum(axis=1) > 0).astype(float)

    # Clear alarms (indices 9–12) — inverse of first 4 breach alarms
    for i in range(4):
        y_alarm[:, N_STATS + i] = 1.0 - y_alarm[:, i]

    return y_alarm


# ─────────────────────────────────────────────
# TRAINING HISTORY
# ─────────────────────────────────────────────

@dataclass
class TrainingHistory:
    train_loss:        List[float] = field(default_factory=list)
    val_loss:          List[float] = field(default_factory=list)
    train_value_loss:  List[float] = field(default_factory=list)
    train_breach_loss: List[float] = field(default_factory=list)
    train_alarm_loss:  List[float] = field(default_factory=list)
    val_value_loss:    List[float] = field(default_factory=list)
    val_breach_loss:   List[float] = field(default_factory=list)
    val_alarm_loss:    List[float] = field(default_factory=list)
    lr_history:        List[float] = field(default_factory=list)
    epoch_times:       List[float] = field(default_factory=list)
    breach_auc:        List[float] = field(default_factory=list)
    alarm_auc:         List[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(np.argmin(self.val_loss)) + 1

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "epoch":            list(range(1, len(self.train_loss)+1)),
            "train_loss":       self.train_loss,
            "val_loss":         self.val_loss,
            "breach_auc":       self.breach_auc,
            "alarm_auc":        self.alarm_auc,
        })


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

class CorrelationTrainer:

    def __init__(self, model: CorrelationLSTM, config: TrainConfig,
                 breach_pos_weight: float = 5.0,
                 alarm_pos_weight:  float = 3.0):
        self.model   = model.to(config.device)
        self.config  = config
        self.history = TrainingHistory()
        self.best_val_loss  = float("inf")
        self.patience_count = 0

        self.criterion = CorrelationLoss(
            w_value=config.w_value,
            w_breach=config.w_breach,
            w_alarm=config.w_alarm,
            breach_pos_weight=breach_pos_weight,
            alarm_pos_weight=alarm_pos_weight,
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    # ── single epoch ──────────────────────────

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool = True
    ) -> Dict[str, float]:

        self.model.train(training)
        total_losses = {"total": 0, "value": 0, "breach": 0, "alarm": 0}
        all_breach_true, all_breach_pred = [], []
        all_alarm_true,  all_alarm_pred  = [], []
        n_batches = 0

        with torch.set_grad_enabled(training):
            for X, y_val, y_breach, y_alarm in loader:
                X        = X.to(self.config.device)
                y_val    = y_val.to(self.config.device)
                y_breach = y_breach.to(self.config.device)
                y_alarm  = y_alarm.to(self.config.device)

                pred_val, pred_breach, pred_alarm, _ = self.model(X)

                loss, breakdown = self.criterion(
                    pred_val, pred_breach, pred_alarm,
                    y_val,    y_breach,    y_alarm
                )

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Track gradient norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    ).item()
                    self.optimizer.step()
                else:
                    grad_norm = 0.0

                for k in total_losses:
                    total_losses[k] += breakdown[k]

                all_breach_true.append(y_breach.cpu().numpy())    # (B, H, 9)
                all_breach_pred.append(pred_breach.detach().cpu().numpy())
                all_alarm_true.append(y_alarm.cpu().numpy())
                all_alarm_pred.append(pred_alarm.detach().cpu().numpy())
                n_batches += 1

        avg = {k: v / n_batches for k, v in total_losses.items()}

        # AUC — per-stat, then macro average
        # Shape: (N_windows * horizon, N_STATS)
        bt = np.concatenate(all_breach_true,  axis=0).reshape(-1, N_STATS)   # (N*H, 9)
        bp = np.concatenate(all_breach_pred,  axis=0).reshape(-1, N_STATS)
        at = np.concatenate(all_alarm_true,   axis=0)
        ap = np.concatenate(all_alarm_pred,   axis=0)

        # Per-stat breach AUC (binarize soft labels at 0.5)
        stat_aucs = []
        for si in range(N_STATS):
            y_true_s = (bt[:, si] > 0.5).astype(int)
            y_pred_s = bp[:, si]
            if y_true_s.sum() < 5:   # not enough positives for AUC
                stat_aucs.append(0.5)
                continue
            try:
                stat_aucs.append(roc_auc_score(y_true_s, y_pred_s))
            except Exception:
                stat_aucs.append(0.5)

        breach_auc = float(np.mean(stat_aucs))
        avg["breach_auc_per_stat"] = stat_aucs

        try:
            alarm_auc = roc_auc_score(at, ap, average="macro")
        except Exception:
            alarm_auc = 0.5

        avg["breach_auc"] = breach_auc
        avg["alarm_auc"]  = alarm_auc
        return avg

    # ── full training loop ────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        progress_callback: Optional[Callable] = None
    ) -> TrainingHistory:

        print(f"\n{'='*60}")
        print(f"  Training on: {self.config.device.upper()}")
        print(f"  Epochs: {self.config.epochs} | Batch: {self.config.batch_size}")
        print(f"  Window: {self.config.window_size} polls | Horizon: {self.config.horizon} polls")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics   = self._run_epoch(val_loader,   training=False)

            # LR scheduler
            self.scheduler.step(val_metrics["total"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history.train_loss.append(train_metrics["total"])
            self.history.val_loss.append(val_metrics["total"])
            self.history.train_value_loss.append(train_metrics["value"])
            self.history.train_breach_loss.append(train_metrics["breach"])
            self.history.train_alarm_loss.append(train_metrics["alarm"])
            self.history.val_value_loss.append(val_metrics["value"])
            self.history.val_breach_loss.append(val_metrics["breach"])
            self.history.val_alarm_loss.append(val_metrics["alarm"])
            self.history.lr_history.append(current_lr)
            self.history.epoch_times.append(time.time() - t0)
            self.history.breach_auc.append(val_metrics["breach_auc"])
            self.history.alarm_auc.append(val_metrics["alarm_auc"])

            # Print progress
            from data_pipeline import ALL_STATS
            per_stat = val_metrics.get("breach_auc_per_stat", [])
            stat_str = ""
            if per_stat:
                stat_str = "  [" + " ".join(
                    f"{s[:4]}:{v:.2f}" for s, v in zip(ALL_STATS, per_stat)
                ) + "]"

            print(
                f"Epoch {epoch:3d}/{self.config.epochs}  "
                f"Loss: {train_metrics['total']:.4f}→{val_metrics['total']:.4f}  "
                f"[val: val={val_metrics['value']:.3f} "
                f"brch={val_metrics['breach']:.3f} "
                f"alrm={val_metrics['alarm']:.3f}]  "
                f"BreachAUC:{val_metrics['breach_auc']:.3f}  "
                f"AlarmAUC:{val_metrics['alarm_auc']:.3f}  "
                f"LR:{current_lr:.2e}  ({time.time()-t0:.1f}s)"
                f"{stat_str}"
            )

            if progress_callback:
                progress_callback(epoch, self.history)

            # Checkpoint best model
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss  = val_metrics["total"]
                self.patience_count = 0
                self._save_checkpoint("best_model.pt")
            else:
                self.patience_count += 1

            # Early stopping
            if self.patience_count >= self.config.patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (patience={self.config.patience})")
                break

        print(f"\n✓ Training complete. Best epoch: {self.history.best_epoch()}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def _save_checkpoint(self, fname: str):
        path = os.path.join(self.config.output_dir, fname)
        torch.save({
            "model_state":    self.model.state_dict(),
            "optimizer_state":self.optimizer.state_dict(),
            "best_val_loss":  self.best_val_loss,
            "history":        asdict(self.history),
            "config":         asdict(self.config),
        }, path)

    def load_best(self):
        path = os.path.join(self.config.output_dir, "best_model.pt")
        ck   = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ck["model_state"])
        print(f"✓ Loaded best model (val_loss={ck['best_val_loss']:.4f})")


# ─────────────────────────────────────────────
# TRAINING ANALYSIS PLOTS
# ─────────────────────────────────────────────

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT1   = "#58a6ff"   # blue
ACCENT2   = "#f78166"   # red/orange
ACCENT3   = "#56d364"   # green
ACCENT4   = "#d2a8ff"   # purple
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
MUTED_COL = "#8b949e"

STAT_COLORS = {
    "avail":    "#56d364",
    "c_util":   "#58a6ff",
    "m_util":   "#d2a8ff",
    "bf_util":  "#ffa657",
    "n_avail":  "#79c0ff",
    "util":     "#f78166",
    "errs":     "#ff7b72",
    "discards": "#ffa657",
    "vol":      "#3fb950",
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED_COL, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.xaxis.label.set_color(MUTED_COL)
    ax.yaxis.label.set_color(MUTED_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def plot_training_analysis(
    history: TrainingHistory,
    corr_matrix: pd.DataFrame,
    causal_graph: Dict,
    val_breach_true: np.ndarray,
    val_breach_pred: np.ndarray,
    val_alarm_true:  np.ndarray,
    val_alarm_pred:  np.ndarray,
    output_dir: str = "outputs"
) -> str:
    """
    Generate comprehensive 6-panel training analysis figure.
    Returns file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 22), facecolor=DARK_BG)
    fig.suptitle(
        "ML Correlation Engine — Training Analysis",
        fontsize=18, fontweight="bold", color=TEXT_COL, y=0.98
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.42, wspace=0.32,
        top=0.95, bottom=0.04, left=0.06, right=0.97
    )

    epochs = list(range(1, len(history.train_loss) + 1))

    # ── Panel 1: Loss curves ──────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, history.train_loss, color=ACCENT1, lw=2, label="Train Total")
    ax1.plot(epochs, history.val_loss,   color=ACCENT2, lw=2, label="Val Total")
    ax1.plot(epochs, history.train_breach_loss, color=ACCENT3, lw=1.5, ls="--", label="Train Breach")
    ax1.plot(epochs, history.val_breach_loss,   color=ACCENT4, lw=1.5, ls="--", label="Val Breach")
    be = history.best_epoch()
    ax1.axvline(be, color=ACCENT3, ls=":", alpha=0.6)
    ax1.text(be + 0.3, max(history.train_loss) * 0.9, f"Best\nEp {be}",
             color=ACCENT3, fontsize=8)
    ax1.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT_COL, framealpha=0.8)
    _style_ax(ax1, "Loss Curves", "Epoch", "Loss")

    # ── Panel 2: AUC curves ───────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, history.breach_auc, color=ACCENT3, lw=2, label="Breach AUC")
    ax2.plot(epochs, history.alarm_auc,  color=ACCENT4, lw=2, label="Alarm AUC")
    ax2.axhline(0.5, color=MUTED_COL, ls=":", alpha=0.5)
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT_COL)
    _style_ax(ax2, "AUC (Validation)", "Epoch", "AUC")

    # ── Panel 3: Correlation matrix ───────────
    ax3 = fig.add_subplot(gs[1, :2])
    cmap = LinearSegmentedColormap.from_list("nw", ["#f78166", CARD_BG, "#58a6ff"])
    im = ax3.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(ALL_STATS)))
    ax3.set_yticks(range(len(ALL_STATS)))
    ax3.set_xticklabels(ALL_STATS, rotation=45, ha="right", fontsize=8, color=TEXT_COL)
    ax3.set_yticklabels(ALL_STATS, fontsize=8, color=TEXT_COL)
    for i in range(len(ALL_STATS)):
        for j in range(len(ALL_STATS)):
            val = corr_matrix.values[i, j]
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if abs(val) > 0.5 else MUTED_COL)
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
    ax3.set_facecolor(CARD_BG)
    ax3.set_title("Stat Cross-Correlation Matrix (lag=0)", fontsize=10,
                  fontweight="bold", color=TEXT_COL, pad=8)

    # ── Panel 4: Causal graph (adjacency heatmap) ─────
    ax4 = fig.add_subplot(gs[1, 2])
    adj = np.zeros((len(ALL_STATS), len(ALL_STATS)))
    for i, s1 in enumerate(ALL_STATS):
        if s1 in causal_graph:
            for s2 in causal_graph[s1]:
                if s2 in ALL_STATS:
                    j = ALL_STATS.index(s2)
                    adj[i, j] = 1
    ax4.imshow(adj, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax4.set_xticks(range(len(ALL_STATS)))
    ax4.set_yticks(range(len(ALL_STATS)))
    ax4.set_xticklabels(ALL_STATS, rotation=45, ha="right", fontsize=7, color=TEXT_COL)
    ax4.set_yticklabels(ALL_STATS, fontsize=7, color=TEXT_COL)
    ax4.set_facecolor(CARD_BG)
    ax4.set_title("Granger Causal Graph\n(row→col)", fontsize=10,
                  fontweight="bold", color=TEXT_COL)

    # ── Panel 5: Per-stat breach AUC ─────────
    ax5 = fig.add_subplot(gs[2, :])

    per_stat_auc = []
    for si, stat in enumerate(ALL_STATS):
        y_t = val_breach_true[:, :, si].ravel()
        y_p = val_breach_pred[:, :, si].ravel()
        try:
            auc = roc_auc_score(y_t, y_p)
        except Exception:
            auc = 0.5
        per_stat_auc.append(auc)

    colors = [STAT_COLORS.get(s, ACCENT1) for s in ALL_STATS]
    bars = ax5.bar(ALL_STATS, per_stat_auc, color=colors, edgecolor=DARK_BG, linewidth=0.5)
    ax5.axhline(0.5,  color=MUTED_COL, ls=":", alpha=0.5, label="Random")
    ax5.axhline(0.85, color=ACCENT3,   ls=":", alpha=0.5, label="Good threshold")
    for bar, val in zip(bars, per_stat_auc):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_COL)
    ax5.set_ylim(0.4, 1.05)
    ax5.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT_COL)
    _style_ax(ax5, "Per-Stat Breach Prediction AUC", "Stat", "AUC")

    # ── Panel 6: LR + Epoch time ──────────────
    ax6a = fig.add_subplot(gs[3, :2])
    ax6b = ax6a.twinx()
    ax6a.plot(epochs, history.lr_history,    color=ACCENT4, lw=2, label="LR")
    ax6b.plot(epochs, history.epoch_times,   color=ACCENT2, lw=1.5, ls="--", label="Epoch Time (s)")
    ax6a.set_yscale("log")
    ax6a.tick_params(colors=MUTED_COL, labelsize=8)
    ax6b.tick_params(colors=MUTED_COL, labelsize=8)
    ax6a.yaxis.label.set_color(ACCENT4)
    ax6b.yaxis.label.set_color(ACCENT2)
    lines1, labels1 = ax6a.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6a.legend(lines1+lines2, labels1+labels2, fontsize=8,
                facecolor=CARD_BG, labelcolor=TEXT_COL)
    _style_ax(ax6a, "Learning Rate & Epoch Time", "Epoch", "LR")
    ax6a.set_facecolor(CARD_BG)
    ax6b.set_facecolor(CARD_BG)
    ax6b.spines[:].set_color(GRID_COL)

    # ── Panel 7: Alarm confusion matrix ───────
    ax7 = fig.add_subplot(gs[3, 2])
    alarm_pred_bin = (val_alarm_pred > 0.5).astype(int)
    alarm_true_bin = val_alarm_true.astype(int)
    short_names = [n.replace("Device ", "Dev ").replace("Utilization", "Util")
                   .replace("Link ", "L ")[:18] for n in ALARM_NAMES]
    cm = confusion_matrix(
        alarm_true_bin.ravel(), alarm_pred_bin.ravel()
    )
    if cm.shape == (2, 2):
        ax7.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax7.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color=TEXT_COL, fontsize=10)
        ax7.set_xticks([0, 1])
        ax7.set_yticks([0, 1])
        ax7.set_xticklabels(["Pred 0", "Pred 1"], color=TEXT_COL, fontsize=8)
        ax7.set_yticklabels(["True 0", "True 1"], color=TEXT_COL, fontsize=8)
    ax7.set_facecolor(CARD_BG)
    ax7.set_title("Alarm Confusion\nMatrix (Aggregated)", fontsize=9,
                  fontweight="bold", color=TEXT_COL)

    path = os.path.join(output_dir, "training_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✓ Training analysis saved → {path}")
    return path


# ─────────────────────────────────────────────
# FULL TRAINING RUN
# ─────────────────────────────────────────────

def run_training(
    X:        np.ndarray,
    y_values: np.ndarray,
    y_breach: np.ndarray,
    corr_matrix: pd.DataFrame,
    causal_graph: Dict,
    config:   Optional[TrainConfig] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[CorrelationLSTM, TrainingHistory, str]:

    if config is None:
        config = TrainConfig()

    # Alarm labels
    y_alarm = build_alarm_labels(y_breach, config.horizon)

    # ── Time-based train/val split (no shuffle — no future leakage)
    n     = len(X)
    split = int(n * (1 - config.val_split))
    X_train, X_val         = X[:split], X[split:]
    yv_train, yv_val       = y_values[:split], y_values[split:]
    yb_train, yb_val       = y_breach[:split], y_breach[split:]
    ya_train, ya_val       = y_alarm[:split],  y_alarm[split:]

    # ── Stratified sampling: keep ALL hard-breach windows + sample normal ones
    # Hard breach = actual threshold crossing (not just proximity)
    # Use y_breach max per window as signal — hard breach produces values close to 1.0
    hard_breach_any = (yb_train > 0.8).any(axis=(1, 2))   # actual threshold crossing
    soft_breach_any = (yb_train > 0.3).any(axis=(1, 2))   # proximity signal
    normal_any      = ~soft_breach_any                      # truly flat windows

    idx_hard   = np.where(hard_breach_any)[0]
    idx_soft   = np.where(soft_breach_any & ~hard_breach_any)[0]
    idx_normal = np.where(normal_any)[0]

    print(f"  Window types — hard breach: {len(idx_hard)} | "
          f"soft/proximity: {len(idx_soft)} | normal: {len(idx_normal)}")

    # Target: 20% hard breach, 40% soft proximity, 40% normal
    max_w        = config.max_train_windows
    n_hard_tgt   = min(len(idx_hard) * 3, int(max_w * 0.20))   # oversample hard 3×
    n_soft_tgt   = min(len(idx_soft),     int(max_w * 0.40))
    n_normal_tgt = min(len(idx_normal),   max_w - n_hard_tgt - n_soft_tgt)

    def sample(idx, n):
        if len(idx) == 0: return np.array([], dtype=int)
        if n >= len(idx):  return idx
        return np.random.choice(idx, size=n, replace=False)

    def oversample(idx, n):
        if len(idx) == 0: return np.array([], dtype=int)
        return np.random.choice(idx, size=n, replace=(n > len(idx)))

    hard_sel   = oversample(idx_hard,   n_hard_tgt)
    soft_sel   = sample(idx_soft,       n_soft_tgt)
    normal_sel = sample(idx_normal,     n_normal_tgt)

    final_idx  = np.concatenate([hard_sel, soft_sel, normal_sel])
    np.random.shuffle(final_idx)

    hard_frac = len(hard_sel) / max(len(final_idx), 1)
    print(f"  After sampling: {len(final_idx)} windows "
          f"| hard={len(hard_sel)} ({hard_frac:.1%}) "
          f"| soft={len(soft_sel)} | normal={len(normal_sel)}")

    X_train  = X_train[final_idx]
    yv_train = yv_train[final_idx]
    yb_train = yb_train[final_idx]
    ya_train = ya_train[final_idx]

    breach_pos_rate = float((yb_train > 0.5).mean())
    dynamic_pos_w   = 5.0   # fixed — soft labels don't need dynamic weighting
    print(f"  Hard breach rate in training set: {breach_pos_rate:.2%}")

    # Subsample val set for speed (13k→3k windows, still representative)
    max_val = 3000
    if len(X_val) > max_val:
        val_step = len(X_val) // max_val
        X_val    = X_val[::val_step][:max_val]
        yv_val   = yv_val[::val_step][:max_val]
        yb_val   = yb_val[::val_step][:max_val]
        ya_val   = ya_val[::val_step][:max_val]
        print(f"  Val set subsampled to {len(X_val)} windows for speed")

    print(f"  Train windows: {len(X_train)} | Val windows: {len(X_val)}")
    print(f"  Features per step: {X.shape[-1]} | Device: {config.device.upper()}")

    # ── CPU thread optimisation
    if config.device == "cpu":
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())

    train_ds = MetricsDataset(X_train, yv_train, yb_train, ya_train)
    val_ds   = MetricsDataset(X_val,   yv_val,   yb_val,   ya_val)

    import platform
    nw = 0 if platform.system() == "Windows" else 2
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size * 2,
                              shuffle=False, num_workers=nw)

    # ── Build model
    from model import build_model
    model = build_model(n_features=X.shape[-1], config={
        "hidden_dim": config.hidden_dim,
        "n_layers":   config.n_layers,
        "horizon":    config.horizon,
        "n_heads":    config.n_heads,
        "dropout":    config.dropout,
    })
    from model import model_summary
    s = model_summary(model)
    print(f"  Model: {s['trainable_params']:,} params | {s['size_mb']} MB")

    trainer = CorrelationTrainer(model, config,
                                 breach_pos_weight=dynamic_pos_w,
                                 alarm_pos_weight=min(10.0, dynamic_pos_w * 0.5))
    history = trainer.train(train_loader, val_loader, progress_callback)
    trainer.load_best()

    # ── Collect val predictions for plots
    model.eval()
    all_bp, all_ap = [], []
    with torch.no_grad():
        for X_b, _, yb, ya in val_loader:
            _, bp, ap, _ = model(X_b.to(config.device))
            all_bp.append(bp.cpu().numpy())
            all_ap.append(ap.cpu().numpy())
    val_breach_pred = np.concatenate(all_bp, axis=0)
    val_alarm_pred  = np.concatenate(all_ap, axis=0)

    plot_path = plot_training_analysis(
        history, corr_matrix, causal_graph,
        yb_val, val_breach_pred,
        ya_val, val_alarm_pred,
        output_dir=config.output_dir
    )
    history.to_df().to_csv(
        os.path.join(config.output_dir, "training_history.csv"), index=False
    )
    return model, history, plot_path