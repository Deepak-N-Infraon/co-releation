"""
main.py
───────
Orchestrator: ties together data pipeline → correlation analysis
              → training → inference.

Usage:
  python main.py --mode train   --device DEV-RTR-1001 --interface Gi0
  python main.py --mode infer   --device DEV-RTR-1001 --interface Gi0
  python main.py --mode analyze --device DEV-RTR-1001 --interface Gi0
  python main.py --mode demo    (uses synthetic data, no DB required)
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import (
    load_data, load_data_from_csv,
    engineer_features, generate_breach_labels,
    build_sliding_windows, StatScaler, ALL_STATS, ALARM_RULES
)
from correlation_analysis import build_correlation_map, query_correlation_map
from model import build_model, model_summary, N_STATS


def _count_breach_episodes(vals: np.ndarray, thresh: float, direction: str) -> int:
    """Count distinct breach episodes (transitions into breach state)."""
    if direction == "above":
        in_breach = vals > thresh
    else:
        in_breach = vals < thresh
    # Count rising edges (False→True transitions)
    return int(np.sum(np.diff(in_breach.astype(int)) == 1))
from trainer import TrainConfig, run_training
from inference import CorrelationInferenceEngine


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DEMO DATA GENERATOR (no DB needed)
# ─────────────────────────────────────────────

def generate_demo_data(n_polls: int = 2000) -> pd.DataFrame:
    """
    Synthesize realistic correlated time-series data for demo.
    Replicates the patterns in your TimescaleDB:
      - util drives errs (lag 2 polls)
      - c_util drives m_util (lag 1 poll)
      - avail=0 → all stats=0
      - breaches at realistic intervals
    """
    np.random.seed(42)
    t = np.arange(n_polls)

    # Base signals with noise
    util     = np.clip(40 + 30*np.sin(t/120) + np.random.randn(n_polls)*8, 0, 100)
    errs     = np.clip(np.roll(util, 2) * 0.6 + np.random.randn(n_polls)*5, 0, 100)
    discards = np.clip(np.roll(util, 3) * 0.5 + np.random.randn(n_polls)*4, 0, 100)
    vol      = np.clip(util * 0.8 + np.random.randn(n_polls)*6, 0, 100)
    n_avail  = np.where(errs > 85, 0, 100).astype(float)
    c_util   = np.clip(35 + 25*np.sin(t/80 + 1) + np.random.randn(n_polls)*7, 0, 100)
    m_util   = np.clip(np.roll(c_util, 1) * 0.75 + 15 + np.random.randn(n_polls)*5, 0, 100)
    bf_util  = np.clip(np.roll(c_util, 2) * 0.65 + 10 + np.random.randn(n_polls)*4, 0, 100)
    avail    = np.where(c_util > 92, 0, 100).astype(float)

    # When avail=0, zero everything
    mask = avail == 0
    for arr in [c_util, m_util, bf_util, util, errs, discards, vol, n_avail]:
        arr[mask] = 0

    idx = pd.date_range("2026-01-01 00:00", periods=n_polls, freq="5min")
    df  = pd.DataFrame({
        "avail": avail, "c_util": c_util, "m_util": m_util, "bf_util": bf_util,
        "n_avail": n_avail, "util": util, "errs": errs, "discards": discards, "vol": vol
    }, index=idx)
    return df


# ─────────────────────────────────────────────
# TRAIN MODE
# ─────────────────────────────────────────────

def mode_train(args):
    print("\n" + "="*65)
    print("  ML CORRELATION ENGINE — TRAINING")
    print("="*65)

    # 1. Load data
    if args.demo:
        print("\n[1/6] Generating demo data (no DB)...")
        metrics_df = generate_demo_data(n_polls=2000)
        events_df  = pd.DataFrame()
    elif args.csv_metrics:
        print(f"\n[1/6] Loading from CSV: {args.csv_metrics}")
        metrics_df, events_df = load_data_from_csv(args.csv_metrics, args.csv_events or "")
    else:
        print(f"\n[1/6] Loading from TimescaleDB: {args.device} / {args.interface}")
        import psycopg2
        conn = psycopg2.connect(args.db_url)
        metrics_df, events_df = load_data(conn, args.device, args.interface)
        conn.close()

    print(f"  ✓ {len(metrics_df)} polls loaded | {metrics_df.index[0]} → {metrics_df.index[-1]}")

    # 2. Correlation analysis
    print("\n[2/6] Running correlation analysis (Granger + Cross-Correlation)...")
    corr_map = build_correlation_map(metrics_df, max_lag=12)

    print(f"  ✓ Found {len(corr_map['pairs'])} significant stat pairs")
    print(f"  ✓ Top causal drivers: {corr_map['top_drivers'][:4]}")

    # Save correlation map
    with open(os.path.join(OUTPUT_DIR, "correlation_map.pkl"), "wb") as f:
        pickle.dump(corr_map, f)

    # Save correlation summary CSV
    pairs_df = pd.DataFrame(corr_map["pairs"])
    pairs_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_pairs.csv"), index=False)
    print(f"  ✓ Correlation map saved")

    print(f"\n  DATA DIAGNOSTIC:")
    for stat in ALL_STATS:
        vals = metrics_df[stat]
        rule = next((r for r in ALARM_RULES if r[0] == stat), None)
        if rule:
            _, direction, thresh, alarm_msg, _ = rule
            if direction == "above":
                in_breach = (vals > thresh).values.astype(int)
            else:
                in_breach = (vals < thresh).values.astype(int)
            n_breach    = int(in_breach.sum())
            n_episodes  = int(np.sum(np.diff(in_breach) == 1))

            # Episode duration analysis
            durations = []
            cur = 0
            for v in in_breach:
                if v == 1:
                    cur += 1
                elif cur > 0:
                    durations.append(cur)
                    cur = 0
            if cur > 0:
                durations.append(cur)

            if durations:
                avg_dur = np.mean(durations)
                max_dur = np.max(durations)
                single_poll = sum(1 for d in durations if d == 1)
                dur_str = (f"avg_dur={avg_dur:.1f}polls  max={max_dur}  "
                           f"single-poll={single_poll}/{len(durations)} "
                           f"({'⚠ all spikes' if single_poll==len(durations) else '✓ sustained'})")
            else:
                dur_str = "no episodes"

            print(f"    {stat:<12} min={vals.min():6.1f} max={vals.max():6.1f} "
                  f"mean={vals.mean():5.1f} | {n_breach:5d} rows ({n_breach/len(vals):.1%}) "
                  f"| {n_episodes} eps | {dur_str}")
    print()

    # 3. Feature engineering
    print("\n[3/6] Engineering features...")
    feat_df = engineer_features(metrics_df)
    print(f"  ✓ Feature matrix: {feat_df.shape} ({feat_df.shape[1]} features/poll)")

    # 4. Build windows
    print("\n[4/6] Building sliding windows...")
    cfg = TrainConfig(
        window_size = args.window  or 24,
        horizon     = args.horizon or 6,
        epochs      = args.epochs  or 60,
        batch_size  = 64,          # smaller batch = more gradient updates per epoch
        lr          = 1e-4,        # conservative — focal loss is sensitive to LR
        hidden_dim  = 64,
        n_layers    = 1,
        n_heads     = 4,
        dropout     = 0.10,        # less dropout — model needs to memorise rare patterns
        patience    = 20,          # more patience
        max_train_windows = 10000,
        output_dir  = OUTPUT_DIR
    )

    label_df = generate_breach_labels(metrics_df, horizon=cfg.horizon)
    X, y_values, y_breach, ts = build_sliding_windows(
        feat_df, label_df,
        window_size=cfg.window_size,
        horizon=cfg.horizon,
        step=1
    )

    # Scale X
    scaler   = StatScaler()
    X_scaled = scaler.fit_transform(X)

    # CRITICAL: also scale y_values into 0-1 range
    # y_values shape: (N, horizon, 9) — scale using same stat min/max
    stat_min   = scaler.min_[:N_STATS]
    stat_range = scaler.range_[:N_STATS]
    y_val_scaled = (y_values - stat_min) / stat_range
    y_val_scaled = np.clip(y_val_scaled, 0, 1).astype(np.float32)

    print(f"  y_values scaled: min={y_val_scaled.min():.3f} max={y_val_scaled.max():.3f}")

    # Save scaler
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 5. Train
    print(f"\n[5/6] Training LSTM Correlation Model...")
    model, history, plot_path = run_training(
        X_scaled, y_val_scaled, y_breach,
        corr_map["correlation_matrix"],
        corr_map["causal_graph"],
        config=cfg
    )

    # 6. Summary
    print(f"\n[6/6] Training Summary")
    print(f"  Best epoch:      {history.best_epoch()}")
    print(f"  Best val loss:   {min(history.val_loss):.4f}")
    print(f"  Best breach AUC: {max(history.breach_auc):.3f}")
    print(f"  Best alarm  AUC: {max(history.alarm_auc):.3f}")
    print(f"\n  📊 Training analysis plot: {plot_path}")
    print(f"  💾 Model checkpoint:        {OUTPUT_DIR}/best_model.pt")
    print(f"  📋 Correlation pairs:       {OUTPUT_DIR}/correlation_pairs.csv")
    print(f"  📈 Training history:        {OUTPUT_DIR}/training_history.csv")


# ─────────────────────────────────────────────
# INFERENCE MODE
# ─────────────────────────────────────────────

def mode_infer(args):
    print("\n" + "="*65)
    print("  ML CORRELATION ENGINE — REAL-TIME INFERENCE")
    print("="*65)

    # Load artifacts
    with open(os.path.join(OUTPUT_DIR, "correlation_map.pkl"), "rb") as f:
        corr_map = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    ck   = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location="cpu")
    cfg  = ck["config"]
    feat = engineer_features(generate_demo_data(30))
    n_features = feat.shape[1]

    model = build_model(n_features=n_features, config={
        "hidden_dim": cfg["hidden_dim"],
        "n_layers":   cfg["n_layers"],
        "horizon":    cfg["horizon"],
        "n_heads":    cfg["n_heads"],
        "dropout":    0.0,
    })
    model.load_state_dict(ck["model_state"])

    engine = CorrelationInferenceEngine(model, scaler, corr_map, cfg)

    # Get live window
    if args.demo:
        raw_window = generate_demo_data(n_polls=engine.window_size + 10).tail(engine.window_size)
    else:
        import psycopg2
        conn = psycopg2.connect(args.db_url)
        raw_window = engine.fetch_live_window(conn, args.device, args.interface)
        conn.close()

    # Run inference
    result = engine.infer(raw_window, args.device, args.interface)

    # Print formatted output
    print(engine.format_result(result))

    # Save JSON
    out_json = engine.to_json(result)
    with open(os.path.join(OUTPUT_DIR, "inference_result.json"), "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"  💾 Inference JSON saved: {OUTPUT_DIR}/inference_result.json")


# ─────────────────────────────────────────────
# ANALYZE MODE — correlation deep-dive
# ─────────────────────────────────────────────

def mode_analyze(args):
    print("\n" + "="*65)
    print("  ML CORRELATION ENGINE — CORRELATION ANALYSIS")
    print("="*65)

    with open(os.path.join(OUTPUT_DIR, "correlation_map.pkl"), "rb") as f:
        corr_map = pickle.load(f)

    pairs_df = pd.DataFrame(corr_map["pairs"])

    print(f"\n{'─'*65}")
    print(f"  TOP CAUSAL DRIVERS (Granger Causality)")
    print(f"{'─'*65}")
    for stat in corr_map["top_drivers"]:
        causes = corr_map["causal_graph"].get(stat, [])
        print(f"  {stat:<12} → affects: {', '.join(causes)}")

    print(f"\n{'─'*65}")
    print(f"  STRONGEST CORRELATIONS (|r| > 0.5)")
    print(f"{'─'*65}")
    strong = pairs_df[pairs_df["abs_correlation"] > 0.5].sort_values(
        "abs_correlation", ascending=False
    ).head(15)
    for _, row in strong.iterrows():
        lag_str = f"{row['lag_minutes']:+.0f} min"
        print(
            f"  {row['driver']:<10} {'→':1} {row['affected']:<10} "
            f"r={row['correlation']:+.3f}  lag={lag_str:>8}  "
            f"{'GRANGER ✓' if row['granger_causes'] else ''}"
        )

    print(f"\n{'─'*65}")
    print(f"  BIDIRECTIONAL QUERY EXAMPLE: util (rising)")
    print(f"{'─'*65}")
    hits = query_correlation_map(corr_map, "util", "rising")
    for h in hits[:6]:
        print(
            f"  util↑ → {h['affected_stat']:<10} will {h['effect']:<5} "
            f"in {h['lag_minutes']:+d} min  "
            f"r={h['correlation']:+.3f}  conf={h['confidence']:.0f}%  {h['strength']}"
        )

    print(f"\n{'─'*65}")
    print(f"  BIDIRECTIONAL QUERY EXAMPLE: util (falling)")
    print(f"{'─'*65}")
    hits = query_correlation_map(corr_map, "util", "falling")
    for h in hits[:6]:
        print(
            f"  util↓ → {h['affected_stat']:<10} will {h['effect']:<5} "
            f"in {h['lag_minutes']:+d} min  "
            f"r={h['correlation']:+.3f}  conf={h['confidence']:.0f}%  {h['strength']}"
        )


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Correlation Engine")
    parser.add_argument("--mode",      choices=["train", "infer", "analyze", "demo"], default="demo")
    parser.add_argument("--device",    default="DEV-RTR-1001")
    parser.add_argument("--interface", default="Gi0")
    parser.add_argument("--db-url",    default="postgresql://user:pass@localhost:5432/netdb")
    parser.add_argument("--csv-metrics", default=None)
    parser.add_argument("--csv-events",  default=None)
    parser.add_argument("--window",    type=int, default=24)
    parser.add_argument("--horizon",   type=int, default=6)
    parser.add_argument("--epochs",    type=int, default=80)
    parser.add_argument("--demo",      action="store_true", default=False)
    args = parser.parse_args()

    if args.mode == "demo":
        args.demo = True
        mode_train(args)
        mode_analyze(args)
        mode_infer(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "infer":
        mode_infer(args)
    elif args.mode == "analyze":
        mode_analyze(args)