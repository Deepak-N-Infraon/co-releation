"""
model.py  — v2
──────────────
Key architectural fixes over v1:

1. Per-stat GRU encoders
   avail/n_avail have 14% breach rate; util/errs have 0.16%.
   A shared LSTM averages their gradients → the 14% stats dominate
   and the 0.16% stats never get learned.
   Fix: each stat gets its own GRU track.

2. Cross-stat attention AFTER per-stat encoding
   Now the model can learn "util rising → errs will follow" without
   the two stats interfering during encoding.

3. Focal loss instead of BCE + pos_weight
   For 0.16% positive rate, focal loss (gamma=2, alpha=0.75) is
   mathematically superior — it down-weights the easy negatives
   instead of up-weighting the positives.

4. Per-stat prediction heads
   Each stat has its own breach head and value head, so the
   easy stats (avail) don't interfere with the hard ones (util).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

# ─────────────────────────────────────────────
ALARM_NAMES = [
    "Device Not Reachable",   "Link Down",
    "Link Utilization High",  "Link Error Rate High",
    "Link Discards Rate High","Link Throughput High",
    "Device CPU Utilization High",
    "Device Memory Utilization High",
    "Device Buffer Utilization High",
    "Device Reachable",       "Link Up",
    "Link Utilization Normal","Link Errors Rate Normal",
]
N_ALARMS        = len(ALARM_NAMES)
N_STATS         = 9
FEATS_PER_STAT  = 8   # raw, roc, rmean, rstd, rmin, rmax, breach_flag, dir


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """
    Focal loss for extreme class imbalance (0.16% positive rate).
    Down-weights easy negatives so the model focuses on hard positives.
    alpha=0.75 → 75% gradient weight on positive class
    gamma=2    → (1-p)^2 modulation factor
    """
    pred   = pred.clamp(1e-6, 1 - 1e-6)
    bce    = F.binary_cross_entropy(pred, target, reduction="none")
    pt     = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1,
                          torch.full_like(pred, alpha),
                          torch.full_like(pred, 1 - alpha))
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


# ─────────────────────────────────────────────
# PER-STAT GRU ENCODER
# ─────────────────────────────────────────────

class StatEncoder(nn.Module):
    """
    One GRU per stat. Encodes its own 8 features over W timesteps.
    Returns a single embedding vector summarising the stat's recent behaviour.
    """
    def __init__(self, feats: int, hidden: int, dropout: float):
        super().__init__()
        self.gru  = nn.GRU(feats, hidden, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, feats)
        out, _ = self.gru(x)              # (B, W, 2H)
        last   = out[:, -1, :]            # (B, 2H) — most recent hidden state
        return self.drop(self.norm(F.gelu(self.proj(last))))  # (B, H)


# ─────────────────────────────────────────────
# CROSS-STAT ATTENTION
# ─────────────────────────────────────────────

class CrossStatAttention(nn.Module):
    """
    Multi-head attention across the N_STATS dimension.
    Learns: 'given util's state, what does that imply for errs?'
    Attention weights become the correlation map for interpretability.
    """
    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N_STATS, H)
        out, w = self.attn(x, x, x)
        return self.norm(x + self.drop(out)), w   # (B, N_STATS, H), (B, N_STATS, N_STATS)


# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────

class CorrelationLSTM(nn.Module):

    def __init__(
        self,
        n_features:     int,
        hidden_dim:     int   = 64,
        n_layers:       int   = 1,      # unused — kept for API compat
        horizon:        int   = 6,
        n_stats:        int   = N_STATS,
        n_alarms:       int   = N_ALARMS,
        n_heads:        int   = 4,
        dropout:        float = 0.15,
        feats_per_stat: int   = FEATS_PER_STAT,
    ):
        super().__init__()
        self.horizon        = horizon
        self.n_stats        = n_stats
        self.hidden_dim     = hidden_dim
        self.feats_per_stat = feats_per_stat

        # Check if features align with per-stat encoding
        self.use_per_stat = (n_features == n_stats * feats_per_stat)
        if not self.use_per_stat:
            # Fallback shared encoder for mismatched feature counts
            print(f"  [model] n_features={n_features} ≠ {n_stats}×{feats_per_stat}="
                  f"{n_stats*feats_per_stat} → using shared encoder")
            self.shared_enc = nn.Sequential(
                nn.Linear(n_features, hidden_dim * n_stats),
                nn.GELU(), nn.Dropout(dropout),
            )
        else:
            # Per-stat GRU encoders
            self.stat_encoders = nn.ModuleList([
                StatEncoder(feats_per_stat, hidden_dim, dropout)
                for _ in range(n_stats)
            ])

        # Cross-stat attention
        self.cross_attn = CrossStatAttention(hidden_dim, n_heads, dropout)

        # Per-stat heads: each gets [own_emb | cross_context] → hidden*2
        head_in = hidden_dim * 2
        self.breach_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, horizon), nn.Sigmoid(),
            ) for _ in range(n_stats)
        ])
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, horizon), nn.Sigmoid(),
            ) for _ in range(n_stats)
        ])

        # Global alarm head
        self.alarm_head = nn.Sequential(
            nn.Linear(hidden_dim * n_stats, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_alarms),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name: nn.init.orthogonal_(p)
                    elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W, F = x.shape

        # ── Encode each stat independently
        if self.use_per_stat:
            embs = []
            for si in range(self.n_stats):
                s = si * self.feats_per_stat
                embs.append(self.stat_encoders[si](x[:, :, s:s+self.feats_per_stat]))
            stat_embs = torch.stack(embs, dim=1)          # (B, N_STATS, H)
        else:
            stat_embs = self.shared_enc(x[:, -1, :]).view(B, self.n_stats, self.hidden_dim)

        # ── Cross-stat attention
        cross_out, attn_w = self.cross_attn(stat_embs)   # (B, N_STATS, H)

        # ── Per-stat predictions
        breach_list, value_list = [], []
        for si in range(self.n_stats):
            inp = torch.cat([stat_embs[:, si, :], cross_out[:, si, :]], dim=-1)
            breach_list.append(self.breach_heads[si](inp))  # (B, horizon)
            value_list.append(self.value_heads[si](inp))    # (B, horizon)

        # (B, N_STATS, horizon) → (B, horizon, N_STATS)
        breach = torch.stack(breach_list, dim=1).permute(0, 2, 1)
        values = torch.stack(value_list,  dim=1).permute(0, 2, 1)

        # ── Alarm prediction
        alarm = self.alarm_head(stat_embs.reshape(B, -1))  # (B, N_ALARMS)

        return values, breach, alarm, attn_w


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

class CorrelationLoss(nn.Module):
    def __init__(self, w_value=0.3, w_breach=3.0, w_alarm=1.5,
                 breach_pos_weight=5.0, alarm_pos_weight=3.0,
                 gamma=2.0, alpha=0.90):   # alpha=0.90 for 0.16% positive rate
        super().__init__()
        self.w_value  = w_value
        self.w_breach = w_breach
        self.w_alarm  = w_alarm
        self.gamma    = gamma
        self.alpha    = alpha
        self.mse      = nn.MSELoss()

    def forward(self, pred_values, pred_breach, pred_alarm,
                true_values, true_breach, true_alarm
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lv = self.mse(pred_values, true_values)
        # Soft breach labels (0.0–1.0 proximity scores) → MSE is correct
        # Focal loss only works on hard 0/1 targets
        lb = self.mse(pred_breach, true_breach)
        # Alarm labels are hard binary → keep focal loss
        la = focal_loss(pred_alarm, true_alarm, self.gamma, self.alpha)
        total = self.w_value * lv + self.w_breach * lb + self.w_alarm * la
        return total, {"total": total.item(), "value": lv.item(),
                       "breach": lb.item(), "alarm": la.item()}


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

def build_model(n_features: int, config: Optional[Dict] = None) -> CorrelationLSTM:
    defaults = dict(hidden_dim=64, n_layers=1, horizon=6, n_heads=4, dropout=0.15)
    if config:
        defaults.update(config)
    return CorrelationLSTM(n_features=n_features, **defaults)

def model_summary(model: CorrelationLSTM) -> Dict:
    t = sum(p.numel() for p in model.parameters())
    r = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": t, "trainable_params": r, "size_mb": round(t*4/1024**2, 2)}