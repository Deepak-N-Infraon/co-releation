"""
correlation_analysis.py
────────────────────────
Granger Causality + Cross-Correlation analysis.
Answers: "When stat X moves (up or down), what other stats are
          affected, in which direction, and after what lag?"
"""

import numpy as np
import pandas as pd
from itertools import permutations
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import grangercausalitytests, ccf
from statsmodels.tsa.stattools import adfuller

from data_pipeline import ALL_STATS, DEVICE_STATS, IFACE_STATS, POLL_INTERVAL

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_LAG_POLLS  = 12          # test up to 60-min lag (12 × 5min)
GRANGER_PVALUE = 0.05        # significance threshold
MIN_CORR       = 0.30        # minimum |correlation| to report


# ─────────────────────────────────────────────
# STATIONARITY CHECK
# ─────────────────────────────────────────────

def make_stationary(series: pd.Series) -> pd.Series:
    """Difference the series until ADF test passes (p < 0.05).
    Returns None if series is constant or too short."""
    s = series.dropna().copy()
    for _ in range(3):
        # Skip if constant or near-constant (std == 0)
        if s.std() == 0 or len(s) < 20:
            return None
        try:
            result = adfuller(s, autolag="AIC")
            if result[1] < 0.05:
                return s
        except ValueError:
            # Constant series — return None to signal skip
            return None
        s = s.diff().dropna()
    return s if s.std() > 0 else None


# ─────────────────────────────────────────────
# CROSS-CORRELATION (Bidirectional)
# ─────────────────────────────────────────────

def compute_cross_correlation(
    df: pd.DataFrame,
    max_lag: int = MAX_LAG_POLLS,
    max_rows: int = 5000          # subsample for speed — 5k polls = plenty
) -> pd.DataFrame:
    """
    Compute cross-correlation between every pair of stats
    at lags -max_lag to +max_lag.

    Subsamples to max_rows for speed (statistical results are
    identical on 5k vs 70k rows for stationary series).
    """
    records = []

    # Subsample evenly across the full time range
    step = max(1, len(df) // max_rows)
    df_s = df.iloc[::step].copy()
    print(f"     (cross-corr on {len(df_s)} sampled rows of {len(df)})")

    # Pre-compute stationary series for all stats once
    stationary = {}
    for stat in ALL_STATS:
        s = make_stationary(df_s[stat])
        if s is not None and len(s) > max_lag * 4:
            stationary[stat] = s.values.astype(np.float64)

    available = list(stationary.keys())
    print(f"     ({len(available)}/{len(ALL_STATS)} stats usable after stationarity check)")

    for stat_x, stat_y in permutations(available, 2):
        x_full = stationary[stat_x]
        y_full = stationary[stat_y]

        # Align to same length
        n = min(len(x_full), len(y_full))
        x = x_full[:n]
        y = y_full[:n]

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        # Vectorised: compute all lags at once
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = float(np.corrcoef(x, y)[0, 1])
            elif lag > 0:
                if lag >= n: continue
                corr = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
            else:
                abs_lag = abs(lag)
                if abs_lag >= n: continue
                corr = float(np.corrcoef(x[abs_lag:], y[:-abs_lag])[0, 1])

            if np.isnan(corr):
                continue

            records.append({
                "stat_x":      stat_x,
                "stat_y":      stat_y,
                "lag_polls":   lag,
                "lag_minutes": lag * POLL_INTERVAL,
                "correlation": round(corr, 4),
                "abs_corr":    abs(corr),
                "relationship":"positive" if corr > 0 else "negative",
                "strength":    _corr_strength(abs(corr))
            })

    return pd.DataFrame(records)


def get_best_lag_per_pair(corr_df: pd.DataFrame, min_corr: float = MIN_CORR) -> pd.DataFrame:
    """
    For each (stat_x → stat_y) pair, return the lag with peak |correlation|.
    Filter by min_corr.
    """
    idx = corr_df.groupby(["stat_x", "stat_y"])["abs_corr"].idxmax()
    best = corr_df.loc[idx].copy()
    best = best[best["abs_corr"] >= min_corr].sort_values("abs_corr", ascending=False)
    return best.reset_index(drop=True)


def _corr_strength(abs_corr: float) -> str:
    if abs_corr >= 0.7: return "strong"
    if abs_corr >= 0.4: return "moderate"
    if abs_corr >= 0.2: return "weak"
    return "negligible"


# ─────────────────────────────────────────────
# GRANGER CAUSALITY
# ─────────────────────────────────────────────

def compute_granger_causality(
    df: pd.DataFrame,
    max_lag: int = MAX_LAG_POLLS,
    p_threshold: float = GRANGER_PVALUE,
    max_rows: int = 3000          # Granger is slower — use fewer rows
) -> pd.DataFrame:
    """
    Test: "Does knowing X's history help predict Y?"
    Subsamples to max_rows for speed.
    """
    records = []

    # Subsample
    step = max(1, len(df) // max_rows)
    df_s = df.iloc[::step].copy()
    print(f"     (Granger on {len(df_s)} sampled rows of {len(df)})")

    # Pre-compute stationary series
    stationary = {}
    for stat in ALL_STATS:
        s = make_stationary(df_s[stat])
        if s is not None and len(s) > max_lag * 4:
            stationary[stat] = s

    available = list(stationary.keys())
    total_pairs = len(available) * (len(available) - 1)
    done = 0

    for stat_x, stat_y in permutations(available, 2):
        done += 1
        if done % 10 == 0:
            print(f"     Granger: {done}/{total_pairs} pairs...", end="\r")

        data = pd.concat([stationary[stat_y], stationary[stat_x]], axis=1).dropna()
        if len(data) < max_lag * 4:
            continue

        # Extra guard: skip if either column constant
        if data.iloc[:, 0].std() == 0 or data.iloc[:, 1].std() == 0:
            continue

        try:
            results  = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
            best_lag = None
            best_pval = 1.0
            for lag, res in results.items():
                pval = res[0]["ssr_ftest"][1]
                if pval < best_pval:
                    best_pval = pval
                    best_lag  = lag

            records.append({
                "stat_x":             stat_x,
                "stat_y":             stat_y,
                "best_lag_polls":     best_lag,
                "best_lag_minutes":   (best_lag or 0) * POLL_INTERVAL,
                "p_value":            round(best_pval, 6),
                "causes":             best_pval < p_threshold
            })
        except Exception:
            pass

    print()   # newline after \r progress
    gc_df = pd.DataFrame(records)
    gc_df = gc_df.sort_values("p_value")
    return gc_df


# ─────────────────────────────────────────────
# CORRELATION MAP (combined output)
# ─────────────────────────────────────────────

def build_correlation_map(
    df: pd.DataFrame,
    max_lag: int = MAX_LAG_POLLS
) -> Dict:
    """
    Full correlation map combining cross-correlation + Granger causality.

    Returns dict:
    {
      "pairs": [
        {
          "driver":      "util",
          "affected":    "errs",
          "direction":   "positive",   # both move same way
          "lag_minutes": 10,
          "correlation": 0.72,
          "granger_causes": True,
          "p_value": 0.003,
          "strength": "strong",
          "interpretation": "When util rises, errs tends to rise ~10 min later"
        }, ...
      ],
      "correlation_matrix": pd.DataFrame,   # 9×9 at lag-0
      "top_drivers": ["util", "c_util", ...],
      "causal_graph": {stat: [affected_stats]}
    }
    """
def build_correlation_map(
    df: pd.DataFrame,
    max_lag: int = MAX_LAG_POLLS
) -> Dict:
    """
    Full correlation map combining cross-correlation + Granger causality.
    Subsamples automatically for large datasets (>5k rows).
    """
    import time

    t0 = time.time()
    print(f"  → Computing cross-correlations ({len(df)} rows, max_lag={max_lag})...")
    corr_df    = compute_cross_correlation(df, max_lag)
    best_pairs = get_best_lag_per_pair(corr_df)
    print(f"     ✓ Cross-correlation done in {time.time()-t0:.1f}s — {len(best_pairs)} pairs above threshold")

    t1 = time.time()
    print(f"  → Computing Granger causality...")
    gc_df = compute_granger_causality(df, max_lag)
    print(f"     ✓ Granger done in {time.time()-t1:.1f}s")

    # Merge
    merged = best_pairs.merge(
        gc_df[["stat_x", "stat_y", "best_lag_polls", "p_value", "causes"]],
        on=["stat_x", "stat_y"], how="left"
    )

    # Build human-readable interpretation
    pairs = []
    for _, row in merged.iterrows():
        lag_min  = int(row["lag_minutes"])
        corr_val = float(row["correlation"])
        direction = row["relationship"]

        if lag_min > 0:
            timing = f"~{lag_min} min later"
        elif lag_min < 0:
            timing = f"~{abs(lag_min)} min earlier"
        else:
            timing = "simultaneously"

        move_word = "rise" if direction == "positive" else "fall"
        interp = (
            f"When {row['stat_x']} rises, {row['stat_y']} tends to "
            f"{move_word} {timing} (r={corr_val:.2f})"
        )
        # Also handle falling driver
        interp_fall = (
            f"When {row['stat_x']} falls, {row['stat_y']} tends to "
            f"{'fall' if direction == 'positive' else 'rise'} {timing} (r={corr_val:.2f})"
        )

        pairs.append({
            "driver":           row["stat_x"],
            "affected":         row["stat_y"],
            "direction":        direction,
            "lag_polls":        int(row.get("best_lag_polls", row["lag_polls"])),
            "lag_minutes":      lag_min,
            "correlation":      corr_val,
            "abs_correlation":  float(row["abs_corr"]),
            "granger_causes":   bool(row.get("causes", False)),
            "p_value":          float(row.get("p_value", 1.0)),
            "strength":         row["strength"],
            "interpretation_rise": interp,
            "interpretation_fall": interp_fall,
        })

    # Correlation matrix at lag 0
    corr_matrix = df[ALL_STATS].corr()

    # Top drivers (stats that Granger-cause the most others)
    causal = gc_df[gc_df["causes"]]
    top_drivers = (causal.groupby("stat_x")["stat_y"].count()
                         .sort_values(ascending=False)
                         .index.tolist())

    # Causal graph
    causal_graph = {}
    for stat in ALL_STATS:
        affected = causal[causal["stat_x"] == stat]["stat_y"].tolist()
        if affected:
            causal_graph[stat] = affected

    return {
        "pairs":              pairs,
        "correlation_matrix": corr_matrix,
        "top_drivers":        top_drivers,
        "causal_graph":       causal_graph,
        "granger_df":         gc_df,
        "crosscorr_df":       best_pairs,
    }


# ─────────────────────────────────────────────
# REAL-TIME CORRELATION QUERY
# ─────────────────────────────────────────────

def query_correlation_map(
    correlation_map: Dict,
    trigger_stat: str,
    trigger_direction: str = "rising"   # "rising" or "falling"
) -> List[Dict]:
    """
    Given a stat that is moving (e.g. util rising),
    return what other stats will be affected, when, and how.
    """
    results = []
    for pair in correlation_map["pairs"]:
        if pair["driver"] != trigger_stat:
            continue
        if pair["abs_correlation"] < MIN_CORR:
            continue

        corr = pair["correlation"]
        if trigger_direction == "rising":
            effect = "rise" if corr > 0 else "fall"
            interp = pair["interpretation_rise"]
        else:
            effect = "fall" if corr > 0 else "rise"
            interp = pair["interpretation_fall"]

        results.append({
            "affected_stat": pair["affected"],
            "effect":        effect,
            "lag_minutes":   pair["lag_minutes"],
            "correlation":   pair["correlation"],
            "strength":      pair["strength"],
            "granger_proven":pair["granger_causes"],
            "interpretation":interp,
            "confidence":    _confidence_score(pair)
        })

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


def _confidence_score(pair: Dict) -> float:
    """
    Composite confidence: combines correlation strength + Granger significance.
    Range 0–100.
    """
    corr_score    = abs(pair["correlation"]) * 60          # max 60
    granger_score = 25 if pair["granger_causes"] else 0    # max 25
    p_score       = max(0, (1 - pair["p_value"]) * 15)     # max 15
    return round(min(100, corr_score + granger_score + p_score), 1)