# ML Correlation Engine — Architecture & Usage Guide

## Overview

A 3-stage ML engine for network telemetry that answers:
1. **When any stat changes (up or down), what other stats are affected and when?**
2. **What is the time lag before correlated stats respond?**
3. **Will any stat breach a threshold and fire an alarm? Which one? When?**

---

## Architecture

```
TimescaleDB
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: CORRELATION ANALYSIS (runs once on historical data)   │
│                                                                 │
│  Cross-Correlation  ──┐                                         │
│  (all 9 stats,        ├──▶  Correlation Map  ──▶  causal_graph  │
│   lags -12 to +12)    │     (72 stat pairs)      lag_per_pair   │
│                       │                                         │
│  Granger Causality  ──┘                                         │
│  (bidirectional,                                                │
│   both rising/falling)                                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: LSTM TRAINING  (sliding window, 5-min polls)          │
│                                                                 │
│  Window [t-W … t]  ──▶  Bidirectional LSTM                      │
│  shape (W=24, F)         + Multi-Head Attention                 │
│                          + 3 output heads:                      │
│                            A) Value forecast  (H, 9)            │
│                            B) Breach probs    (H, 9)   sigmoid  │
│                            C) Alarm probs     (13,)    sigmoid  │
│                                                                 │
│  Sliding step = 1 poll (5 min)                                  │
│  Window:  24 polls = 2 hr lookback                              │
│  Horizon:  6 polls = 30 min forecast                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: REAL-TIME INFERENCE                                   │
│                                                                 │
│  Live poll (T)  ──▶  Rolling window buffer (last 24 polls)      │
│                          │                                      │
│                          ▼                                      │
│                     Feature engineering                         │
│                     (ROC, rolling mean/std, breach flags)       │
│                          │                                      │
│                          ▼                                      │
│                     Trained LSTM                                │
│                          │                                      │
│              ┌───────────┼────────────┐                         │
│              ▼           ▼            ▼                         │
│         Forecast    Breach prob   Alarm prob                    │
│         (9 stats,   (9 stats ×   (13 alarms)                    │
│          30 min)     30 min)                                    │
│              │           │            │                         │
│              └───────────┴────────────┘                         │
│                          │                                      │
│                   Correlation map lookup                        │
│                   (which stats follow? when?)                   │
│                          │                                      │
│                   InferenceResult JSON                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sliding Window — Exact Mechanics

```
Poll interval = 5 minutes
Window size W = 24 polls  = 2 hours lookback
Horizon     H =  6 polls  = 30 minutes forecast
Step            1 poll    = 5 minutes

Training batches:
  Batch  1:  polls [0  … 23]  →  labels [24 … 29]
  Batch  2:  polls [1  … 24]  →  labels [25 … 30]
  Batch  3:  polls [2  … 25]  →  labels [26 … 31]
  ...
  Batch  N:  polls [N  … N+23] → labels [N+24 … N+29]

Inference (real-time):
  On new poll at T → take last 24 polls [T-23 … T]
                   → run model → get forecast for [T+1 … T+6]
                   → each new poll slides the window by 1
```

---

## Feature Vector (per timestep)

Each of the 9 stats contributes 7 features:

| Feature        | Description                          |
|----------------|--------------------------------------|
| raw value      | stat at time t                       |
| roc            | rate of change (delta / 5 min)       |
| rmean          | rolling mean (last 6 polls = 30 min) |
| rstd           | rolling std dev                      |
| rmin           | rolling min                          |
| rmax           | rolling max                          |
| breach flag    | 1 if currently breached              |
| direction      | +1 rising / -1 falling / 0 stable   |

Total features = 9 stats × 8 = **72 features per poll**

---

## Correlation Map Output

For every pair (stat_x → stat_y), 72 directed pairs total:

```json
{
  "driver":           "util",
  "affected":         "errs",
  "direction":        "positive",
  "lag_minutes":      10,
  "correlation":      0.72,
  "granger_causes":   true,
  "p_value":          0.003,
  "strength":         "strong",
  "interpretation_rise": "When util rises, errs tends to rise ~10 min later (r=0.72)",
  "interpretation_fall": "When util falls, errs tends to fall ~10 min later (r=0.72)"
}
```

Both rising and falling are covered automatically via the `direction` field.

---

## Inference Output (per poll)

```json
{
  "overall_health": "degrading",
  "overall_score": 62.5,
  "active_breaches": ["Link Utilization High"],
  "predicted_alarms": [
    { "alarm_msg": "Link Error Rate High", "probability": 0.84, "severity": "critical" }
  ],
  "stat_predictions": [
    {
      "stat": "util",
      "current": 83.2,
      "predicted_values": [85.1, 87.3, 88.0, 86.5, 84.2, 81.0],
      "breach_probability": [0.91, 0.93, 0.95, 0.92, 0.88, 0.79],
      "max_breach_prob": 0.95,
      "time_to_breach_min": null,
      "alarm_msg": "Link Utilization High",
      "severity": "critical",
      "confidence": 87.3
    }
  ],
  "correlation_alerts": [
    {
      "trigger_stat": "util",
      "trigger_direction": "rising",
      "affected_stats": [
        { "affected_stat": "errs", "effect": "rise", "lag_minutes": 10, "confidence": 82.5 }
      ],
      "interpretation": "util is rising (slope=3.2/poll). Expected downstream: errs, discards"
    }
  ]
}
```

---

## Quick Start

### Train (demo, no DB)
```bash
cd ml_engine
pip install torch numpy pandas scikit-learn statsmodels matplotlib seaborn psycopg2
python main.py --mode demo
```

### Train (from TimescaleDB)
```bash
python main.py --mode train \
  --db-url "postgresql://user:pass@localhost:5432/netdb" \
  --device DEV-RTR-1001 \
  --interface Gi0 \
  --window 24 \
  --horizon 6 \
  --epochs 100
```

### Real-time inference
```bash
python main.py --mode infer \
  --db-url "postgresql://user:pass@localhost:5432/netdb" \
  --device DEV-RTR-1001 \
  --interface Gi0
```

### Correlation deep-dive
```bash
python main.py --mode analyze
```

---

## Output Files

| File                        | Description                                 |
|-----------------------------|---------------------------------------------|
| `outputs/best_model.pt`     | PyTorch model checkpoint                    |
| `outputs/scaler.pkl`        | Fitted MinMax scaler                        |
| `outputs/correlation_map.pkl`| Granger + cross-correlation analysis        |
| `outputs/correlation_pairs.csv` | Human-readable correlation table        |
| `outputs/training_analysis.png` | 6-panel training analysis plot          |
| `outputs/training_history.csv`  | Loss / AUC per epoch                    |
| `outputs/inference_result.json` | Latest inference output                 |

---

## Alarm Coverage

| Alarm                          | Stat      | Direction | Threshold |
|--------------------------------|-----------|-----------|-----------|
| Device Not Reachable           | avail     | below     | < 99      |
| Device Reachable (clear)       | avail     | equal     | = 100     |
| Link Down                      | n_avail   | below     | < 99      |
| Link Up (clear)                | n_avail   | equal     | = 100     |
| Link Utilization High          | util      | above     | > 80      |
| Link Utilization Normal (clear)| util      | below/eq  | ≤ 80      |
| Link Error Rate High           | errs      | above     | > 80      |
| Link Discards Rate High        | discards  | above     | > 80      |
| Link Throughput High           | vol       | above     | > 80      |
| Device CPU Utilization High    | c_util    | above     | > 80      |
| Device Memory Utilization High | m_util    | above     | > 80      |
| Device Buffer Utilization High | bf_util   | above     | > 80      |

---

## Key Design Decisions

### Why Bidirectional LSTM?
Forward pass captures "buildup" patterns (stats climbing toward breach).
Backward pass captures "recovery" patterns (stats falling after clear alarm).
Together they model both alarm and clear event trajectories.

### Why Multi-Head Attention?
Network stats have non-uniform temporal dependencies — a spike 45 min ago
may matter more than the last 5 min. Attention learns which past polls
to weight most for each prediction.

### Why Granger + LSTM (not just LSTM)?
Granger causality gives you a **statistical, interpretable** causal map
that doesn't depend on model training. The LSTM adds **non-linear**,
**multi-step** forecasting on top. Together:
- Granger → "util Granger-causes errs at lag 2 polls" (interpretable)
- LSTM    → "given current trajectory, errs will breach in 10 min" (predictive)

### Why Sliding Window Step = 1 Poll?
Maximum data efficiency. With 2000 polls you get ~1975 training windows
instead of ~83 (if step = window size). This is critical for rare events
like breaches which may only occur a few dozen times in the dataset.



