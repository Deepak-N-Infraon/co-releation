"""
data_pipeline.py
────────────────
Extracts time-aligned metrics from TimescaleDB and builds
sliding-window training tensors for the correlation engine.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DEVICE_STATS   = ["avail", "c_util", "m_util", "bf_util"]
IFACE_STATS    = ["n_avail", "util", "errs", "discards", "vol"]
ALL_STATS      = DEVICE_STATS + IFACE_STATS          # 9 total
POLL_INTERVAL  = 5                                    # minutes
BREACH_THRESH  = 80.0
AVAIL_THRESH   = 99.0                                 # avail/n_avail breach < 99

# Alarm definitions: (stat, direction, threshold, alarm_msg, clear_msg)
ALARM_RULES = [
    ("avail",   "below", AVAIL_THRESH, "Device Not Reachable",          "Device Reachable"),
    ("n_avail", "below", AVAIL_THRESH, "Link Down",                     "Link Up"),
    ("util",    "above", BREACH_THRESH,"Link Utilization High",         "Link Utilization Normal"),
    ("errs",    "above", BREACH_THRESH,"Link Error Rate High",          "Link Errors Rate Normal"),
    ("discards","above", BREACH_THRESH,"Link Discards Rate High",       "Link Discards Rate Normal"),
    ("vol",     "above", BREACH_THRESH,"Link Throughput High",          "Link Throughput Normal"),
    ("c_util",  "above", BREACH_THRESH,"Device CPU Utilization High",   "Device CPU Utilization Normal"),
    ("m_util",  "above", BREACH_THRESH,"Device Memory Utilization High","Device Memory Utilization Normal"),
    ("bf_util", "above", BREACH_THRESH,"Device Buffer Utilization High","Device Buffer Utilization Normal"),
]


# ─────────────────────────────────────────────
# TIMESCALEDB EXTRACTION QUERIES
# ─────────────────────────────────────────────

EXTRACT_QUERY = """
WITH time_series AS (
    SELECT generate_series(
        (SELECT MIN(ts) FROM metrics_device),
        (SELECT MAX(ts) FROM metrics_device),
        INTERVAL '5 minutes'
    ) AS ts
),
device_metrics AS (
    SELECT
        ts,
        device_name,
        avail,
        c_util,
        m_util,
        bf_util
    FROM metrics_device
    WHERE device_name = %(device)s
),
iface_metrics AS (
    SELECT
        ts,
        device_name,
        interface_name,
        n_avail,
        util,
        errs,
        discards,
        vol
    FROM metrics_interface
    WHERE device_name = %(device)s
      AND interface_name = %(interface)s
)
SELECT
    t.ts,
    COALESCE(d.avail,    0)::float  AS avail,
    COALESCE(d.c_util,   0)::float  AS c_util,
    COALESCE(d.m_util,   0)::float  AS m_util,
    COALESCE(d.bf_util,  0)::float  AS bf_util,
    COALESCE(i.n_avail,  0)::float  AS n_avail,
    COALESCE(i.util,     0)::float  AS util,
    COALESCE(i.errs,     0)::float  AS errs,
    COALESCE(i.discards, 0)::float  AS discards,
    COALESCE(i.vol,      0)::float  AS vol
FROM time_series t
LEFT JOIN device_metrics d ON d.ts = t.ts
LEFT JOIN iface_metrics  i ON i.ts = t.ts
ORDER BY t.ts;
"""

EVENTS_QUERY = """
SELECT
    ts,
    device_name,
    interface_name,
    alarm_msg,
    stat,
    is_cleared
FROM events
WHERE device_name = %(device)s
ORDER BY ts;
"""


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────

def load_data(conn, device: str, interface: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load aligned metrics + events from TimescaleDB.
    Returns (metrics_df, events_df)
    """
    metrics_df = pd.read_sql(
        EXTRACT_QUERY,
        conn,
        params={"device": device, "interface": interface},
        parse_dates=["ts"],
        index_col="ts"
    )
    events_df = pd.read_sql(
        EVENTS_QUERY,
        conn,
        params={"device": device},
        parse_dates=["ts"]
    )
    logger.info(f"Loaded {len(metrics_df)} metric rows, {len(events_df)} events")
    return metrics_df, events_df


def load_data_from_csv(metrics_csv: str, events_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load from CSV files (dev/testing without live DB)."""
    metrics_df = pd.read_csv(metrics_csv, parse_dates=["ts"], index_col="ts")
    events_df  = pd.read_csv(events_csv,  parse_dates=["ts"])
    return metrics_df, events_df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
    """
    Add derived features per stat:
      - rate_of_change  (delta / poll_interval)
      - rolling_mean    (last window_size polls)
      - rolling_std     (last window_size polls)
      - rolling_min/max
      - breach_flag     (1 if stat breached threshold this poll)
      - direction       (+1 rising, -1 falling, 0 stable)
    """
    feat = df[ALL_STATS].copy()

    for stat in ALL_STATS:
        thresh = AVAIL_THRESH if stat in ("avail", "n_avail") else BREACH_THRESH
        direction = "below" if stat in ("avail", "n_avail") else "above"

        feat[f"{stat}_roc"]    = feat[stat].diff() / POLL_INTERVAL
        feat[f"{stat}_rmean"]  = feat[stat].rolling(window_size).mean()
        feat[f"{stat}_rstd"]   = feat[stat].rolling(window_size).std().fillna(0)
        feat[f"{stat}_rmin"]   = feat[stat].rolling(window_size).min()
        feat[f"{stat}_rmax"]   = feat[stat].rolling(window_size).max()

        if direction == "above":
            feat[f"{stat}_breach"] = (feat[stat] > thresh).astype(float)
        else:
            feat[f"{stat}_breach"] = (feat[stat] < thresh).astype(float)

        feat[f"{stat}_dir"] = np.sign(feat[f"{stat}_roc"]).fillna(0)

    feat = feat.fillna(0)
    return feat


# ─────────────────────────────────────────────
# BREACH LABEL GENERATION
# ─────────────────────────────────────────────

def generate_breach_labels(df: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """
    Trend-aware breach label generation.

    For each timestep t, the label is NOT just "will it cross 80 in H steps"
    (useless for single-poll spikes) but a SOFT score based on:
      1. Hard breach: stat crosses threshold in horizon (binary, weight 1.0)
      2. Trend breach: stat is rising fast AND within 20% of threshold (soft, weight 0.7)
      3. Proximity: how close is stat to threshold right now (0-1 continuous)

    Combined label = max(hard, soft_trend, proximity * 0.5)
    This gives the model meaningful gradient signal even when hard breaches are rare.
    """
    labels      = pd.DataFrame(index=df.index)
    breach_rates = {}

    for stat, direction, thresh, alarm_msg, clear_msg in ALARM_RULES:
        if stat not in df.columns:
            continue
        series = df[stat].values.astype(np.float64)
        n      = len(series)

        # ── 1. Hard breach label (original)
        hard = np.zeros(n, dtype=np.float32)
        for h in range(1, horizon + 1):
            future = np.empty(n, dtype=np.float64)
            future[:] = np.nan
            future[:n-h] = series[h:]
            if direction == "above":
                mask = (future > thresh).astype(np.float32)
            else:
                mask = (future < thresh).astype(np.float32)
            hard = np.maximum(hard, np.where(np.isnan(future), 0.0, mask).astype(np.float32))
        hard[-horizon:] = 0.0

        # ── 2. Proximity label — only meaningful when CLOSE to threshold
        # Use a tight window: only score >0 when within 15% of threshold
        # e.g. thresh=80: val>=68 (85%) starts scoring, val=79 scores ~1.0
        if direction == "above":
            near_zone  = thresh * 0.85         # start scoring at 85% of threshold
            proximity  = np.clip(
                (series - near_zone) / (thresh - near_zone), 0.0, 1.0
            ).astype(np.float32)
        else:
            # For avail/n_avail (thresh=99, breach=below)
            near_zone  = thresh * 1.05          # start scoring when 5% above threshold
            proximity  = np.clip(
                (near_zone - series) / near_zone, 0.0, 1.0
            ).astype(np.float32)

        # ── 3. Trend label — rising fast AND within 20% of threshold
        roc = np.zeros(n, dtype=np.float32)
        roc[1:] = np.diff(series).astype(np.float32)
        rolling_roc = pd.Series(roc).rolling(6, min_periods=1).mean().values.astype(np.float32)

        if direction == "above":
            near_thresh  = (series > thresh * 0.80).astype(np.float32)  # within 20%
            rising_fast  = np.clip(rolling_roc / 3.0, 0.0, 1.0)
            trend_label  = (near_thresh * rising_fast).astype(np.float32)
        else:
            near_thresh  = (series < thresh * 1.05).astype(np.float32)
            falling_fast = np.clip(-rolling_roc / 3.0, 0.0, 1.0)
            trend_label  = (near_thresh * falling_fast).astype(np.float32)

        # ── Combined soft label
        soft_label = np.maximum(
            hard,
            np.maximum(
                trend_label * 0.70,
                proximity   * 0.40
            )
        ).astype(np.float32)

        labels[f"{stat}_will_breach"] = soft_label
        breach_rates[stat] = float((soft_label > 0.3).mean())

    # Diagnostics
    print(f"  Breach label rates (soft labels, threshold>0.3):")
    for stat, rate in breach_rates.items():
        bar = "█" * int(rate * 40)
        print(f"    {stat:<12} {rate:6.2%}  {bar}")
    avg = np.mean(list(breach_rates.values()))
    print(f"  ✓ Average meaningful breach signal: {avg:.2%}")

    return labels


# ─────────────────────────────────────────────
# SLIDING WINDOW BUILDER
# ─────────────────────────────────────────────

def build_sliding_windows(
    features:    pd.DataFrame,
    labels:      pd.DataFrame,
    window_size: int = 24,
    horizon:     int = 6,
    step:        int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Build sliding window tensors.

    X        (N, W, F)  — feature window
    y_values (N, H, 9)  — future raw stat values (scaled later in main)
    y_breach (N, H, 9)  — per-step breach flag (1 if that step's stat is breached)
    """
    feat_arr = features.values.astype(np.float32)
    stat_arr = features[ALL_STATS].values.astype(np.float32)
    n        = len(feat_arr)

    # Build breach targets from SOFT labels (not raw threshold crossing)
    breach_soft = np.zeros((n, len(ALL_STATS)), dtype=np.float32)
    for si, stat in enumerate(ALL_STATS):
        col = f"{stat}_will_breach"
        if col in labels.columns:
            breach_soft[:, si] = labels[col].values.astype(np.float32)

    rates = (breach_soft > 0.3).mean(axis=0)
    print(f"  Per-stat meaningful breach signal (>0.3):")
    for si, stat in enumerate(ALL_STATS):
        print(f"    {stat:<12} {rates[si]:6.2%}")

    total_windows = (n - window_size - horizon + 1)
    if total_windows <= 0:
        raise ValueError(f"Not enough data: {n} rows for window={window_size}+horizon={horizon}")
    indices = np.arange(0, total_windows, step)

    X        = np.stack([feat_arr[i : i+window_size]           for i in indices])
    y_values = np.stack([stat_arr[i+window_size : i+window_size+horizon] for i in indices])
    # For per-step breach targets: use the soft label AT each future timestep
    y_breach = np.stack([breach_soft[i+window_size : i+window_size+horizon] for i in indices])
    ts       = [features.index[i+window_size] for i in indices]

    print(f"  Windows built: {len(X)} | X:{X.shape} | y_val:{y_values.shape} | y_breach:{y_breach.shape}")
    pos_rate = (y_breach > 0.3).mean()
    print(f"  y_breach signal rate (>0.3): {pos_rate:.2%} "
          f"{'✓ good' if pos_rate > 0.02 else '⚠ low — using soft labels helps'}")

    return (X.astype(np.float32),
            y_values.astype(np.float32),
            y_breach.astype(np.float32),
            ts)


# ─────────────────────────────────────────────
# NORMALIZER
# ─────────────────────────────────────────────

class StatScaler:
    """Min-max scaler fitted on training data only."""

    def __init__(self):
        self.min_   = None
        self.max_   = None
        self.range_ = None

    def fit(self, X: np.ndarray):
        # X shape: (n_samples, window, features)
        flat = X.reshape(-1, X.shape[-1])
        self.min_   = flat.min(axis=0)
        self.max_   = flat.max(axis=0)
        self.range_ = np.where(
            (self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_
        )
        # Diagnostics
        scaled_check = (flat - self.min_) / self.range_
        print(f"  Scaler fitted: global min={scaled_check.min():.3f} "
              f"max={scaled_check.max():.3f} mean={scaled_check.mean():.3f}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform_stats(self, arr: np.ndarray, n_stats: int = 9) -> np.ndarray:
        scale_min   = self.min_[:n_stats]
        scale_range = self.range_[:n_stats]
        return arr * scale_range + scale_min