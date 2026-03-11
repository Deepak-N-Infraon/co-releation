"""
inference.py
────────────
Real-time inference engine.
Input:  latest W polls from TimescaleDB (rolling window)
Output: correlation map + breach predictions + alarm forecasts + severity scores
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from data_pipeline import (
    ALL_STATS, DEVICE_STATS, IFACE_STATS,
    ALARM_RULES, POLL_INTERVAL, engineer_features, StatScaler
)
from model import CorrelationLSTM, ALARM_NAMES, N_STATS
from correlation_analysis import query_correlation_map


# ─────────────────────────────────────────────
# REAL-TIME INFERENCE QUERY (TimescaleDB)
# ─────────────────────────────────────────────

REALTIME_QUERY = """
SELECT
    t.ts,
    COALESCE(d.avail,    0)::float AS avail,
    COALESCE(d.c_util,   0)::float AS c_util,
    COALESCE(d.m_util,   0)::float AS m_util,
    COALESCE(d.bf_util,  0)::float AS bf_util,
    COALESCE(i.n_avail,  0)::float AS n_avail,
    COALESCE(i.util,     0)::float AS util,
    COALESCE(i.errs,     0)::float AS errs,
    COALESCE(i.discards, 0)::float AS discards,
    COALESCE(i.vol,      0)::float AS vol
FROM (
    SELECT generate_series(
        NOW() - INTERVAL '{window_minutes} minutes',
        NOW(),
        INTERVAL '5 minutes'
    ) AS ts
) t
LEFT JOIN metrics_device d
    ON d.ts = t.ts AND d.device_name = %(device)s
LEFT JOIN metrics_interface i
    ON i.ts = t.ts
    AND i.device_name = %(device)s
    AND i.interface_name = %(interface)s
ORDER BY t.ts;
"""


# ─────────────────────────────────────────────
# INFERENCE OUTPUT STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class StatPrediction:
    stat_name:          str
    current_value:      float
    predicted_values:   List[float]       # T+1 to T+H
    breach_probability: List[float]       # per step
    max_breach_prob:    float
    steps_to_breach:    Optional[int]     # None if no breach predicted
    time_to_breach_min: Optional[int]
    alarm_msg:          str
    severity:           str               # low / medium / high / critical
    confidence:         float


@dataclass
class CorrelationAlert:
    trigger_stat:       str
    trigger_direction:  str               # rising / falling
    affected_stats:     List[Dict]
    lag_minutes:        int
    confidence:         float
    interpretation:     str


@dataclass
class InferenceResult:
    device_name:        str
    interface_name:     str
    timestamp:          datetime
    stat_predictions:   List[StatPrediction]
    correlation_alerts: List[CorrelationAlert]
    active_breaches:    List[str]         # stats currently breached
    predicted_alarms:   List[Dict]        # alarms expected in horizon
    overall_health:     str               # healthy / degrading / critical
    overall_score:      float             # 0–100 (100 = perfect)
    attention_map:      Optional[np.ndarray]  # (window, window) attention weights


# ─────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────

class CorrelationInferenceEngine:

    def __init__(
        self,
        model:           CorrelationLSTM,
        scaler:          StatScaler,
        correlation_map: Dict,
        config:          Dict,
        device:          str = "cpu"
    ):
        self.model           = model.to(device).eval()
        self.scaler          = scaler
        self.correlation_map = correlation_map
        self.config          = config
        self.device          = device
        self.window_size     = config.get("window_size", 24)
        self.horizon         = config.get("horizon", 6)
        self.feature_cols    = config.get("feature_cols", ALL_STATS)

    # ── Load model from checkpoint ────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        correlation_map: Dict,
        n_features:      int
    ) -> "CorrelationInferenceEngine":
        from model import build_model

        ck = torch.load(checkpoint_path, map_location="cpu")
        cfg = ck["config"]

        model = build_model(n_features=n_features, config={
            "hidden_dim": cfg["hidden_dim"],
            "n_layers":   cfg["n_layers"],
            "horizon":    cfg["horizon"],
            "n_heads":    cfg["n_heads"],
            "dropout":    cfg["dropout"],
        })
        model.load_state_dict(ck["model_state"])

        scaler = StatScaler()
        if "scaler" in ck:
            scaler.min_   = ck["scaler"]["min_"]
            scaler.max_   = ck["scaler"]["max_"]
            scaler.range_ = ck["scaler"]["range_"]

        return cls(model, scaler, correlation_map, cfg)

    # ── Fetch live window from DB ─────────────

    def fetch_live_window(
        self, conn, device: str, interface: str
    ) -> pd.DataFrame:
        window_minutes = self.window_size * POLL_INTERVAL
        query = REALTIME_QUERY.format(window_minutes=window_minutes)
        df = pd.read_sql(
            query, conn,
            params={"device": device, "interface": interface},
            parse_dates=["ts"],
            index_col="ts"
        )
        return df

    # ── Core inference ────────────────────────

    def infer(
        self,
        raw_window: pd.DataFrame,           # (W, 9) raw stats
        device_name: str    = "DEV-RTR-1001",
        interface_name: str = "Gi0"
    ) -> InferenceResult:
        """
        Main inference call.
        raw_window: DataFrame with columns = ALL_STATS, last W rows.
        """
        # 1. Engineer features
        feat_df    = engineer_features(raw_window)
        feat_arr   = feat_df.values.astype(np.float32)

        if len(feat_arr) < self.window_size:
            raise ValueError(
                f"Need {self.window_size} polls, got {len(feat_arr)}"
            )

        # Use last W rows
        window     = feat_arr[-self.window_size:]
        window_3d  = window[np.newaxis, :, :]                  # (1, W, F)

        if self.scaler.min_ is not None:
            window_scaled = self.scaler.transform(window_3d)
        else:
            window_scaled = window_3d

        X_tensor = torch.tensor(window_scaled, dtype=torch.float32).to(self.device)

        # 2. Model forward pass
        with torch.no_grad():
            pred_values, pred_breach, pred_alarm, attn_weights = self.model(X_tensor)

        pred_values = pred_values.cpu().numpy()[0]    # (H, 9)
        pred_breach = pred_breach.cpu().numpy()[0]    # (H, 9)
        pred_alarm  = pred_alarm.cpu().numpy()[0]     # (13,)
        attn_w      = attn_weights.cpu().numpy()[0]   # (W, W)

        # Inverse-scale predicted values
        if self.scaler.min_ is not None:
            pred_values = self.scaler.inverse_transform_stats(pred_values)

        # 3. Current stat values
        current = raw_window[ALL_STATS].iloc[-1].values

        # 4. Active breaches
        active_breaches = self._check_active_breaches(current)

        # 5. Per-stat predictions
        stat_preds = self._build_stat_predictions(
            current, pred_values, pred_breach
        )

        # 6. Correlation alerts (what is currently trending?)
        corr_alerts = self._build_correlation_alerts(raw_window)

        # 7. Predicted alarms
        predicted_alarms = self._build_alarm_predictions(pred_alarm)

        # 8. Overall health
        health, score = self._compute_health(active_breaches, stat_preds, pred_alarm)

        return InferenceResult(
            device_name=device_name,
            interface_name=interface_name,
            timestamp=datetime.now(),
            stat_predictions=stat_preds,
            correlation_alerts=corr_alerts,
            active_breaches=active_breaches,
            predicted_alarms=predicted_alarms,
            overall_health=health,
            overall_score=score,
            attention_map=attn_w
        )

    # ── Helper builders ───────────────────────

    def _check_active_breaches(self, current: np.ndarray) -> List[str]:
        breaches = []
        for stat, direction, thresh, alarm_msg, _ in ALARM_RULES:
            si = ALL_STATS.index(stat)
            val = current[si]
            if direction == "above" and val > thresh:
                breaches.append(alarm_msg)
            elif direction == "below" and val < thresh:
                breaches.append(alarm_msg)
        return breaches

    def _build_stat_predictions(
        self,
        current: np.ndarray,
        pred_values: np.ndarray,
        pred_breach: np.ndarray
    ) -> List[StatPrediction]:

        results = []
        for i, stat in enumerate(ALL_STATS):
            rule = next((r for r in ALARM_RULES if r[0] == stat), None)
            if rule is None:
                continue
            _, direction, thresh, alarm_msg, _ = rule

            curr_val      = float(current[i])
            pred_vals     = [round(float(v), 2) for v in pred_values[:, i]]
            breach_probs  = [round(float(p), 3) for p in pred_breach[:, i]]
            max_breach    = max(breach_probs)

            # Steps to first likely breach
            steps_to_breach = None
            time_to_breach  = None
            for step, prob in enumerate(breach_probs):
                if prob > 0.5:
                    steps_to_breach = step + 1
                    time_to_breach  = steps_to_breach * POLL_INTERVAL
                    break

            # Severity
            if max_breach >= 0.85:   severity = "critical"
            elif max_breach >= 0.65: severity = "high"
            elif max_breach >= 0.40: severity = "medium"
            else:                    severity = "low"

            # Confidence: based on recent trend consistency
            confidence = self._trend_confidence(current, i, pred_vals)

            results.append(StatPrediction(
                stat_name=stat,
                current_value=curr_val,
                predicted_values=pred_vals,
                breach_probability=breach_probs,
                max_breach_prob=round(max_breach, 3),
                steps_to_breach=steps_to_breach,
                time_to_breach_min=time_to_breach,
                alarm_msg=alarm_msg,
                severity=severity,
                confidence=confidence
            ))
        return results

    def _trend_confidence(
        self, current: np.ndarray, stat_idx: int, pred_vals: List[float]
    ) -> float:
        """Confidence based on how monotonic and strong the trend is."""
        vals = [float(current[stat_idx])] + pred_vals
        diffs = np.diff(vals)
        if len(diffs) == 0:
            return 50.0
        # Monotonicity score
        same_sign = np.sum(np.sign(diffs) == np.sign(diffs[0])) / len(diffs)
        magnitude = min(abs(np.mean(diffs)) / 5.0, 1.0)   # normalise by 5 units/step
        return round((same_sign * 60 + magnitude * 40), 1)

    def _build_correlation_alerts(
        self, raw_window: pd.DataFrame
    ) -> List[CorrelationAlert]:
        """
        Detect which stats are trending and query correlation map
        for downstream effects.
        """
        alerts = []
        recent = raw_window[ALL_STATS].tail(6)

        for stat in ALL_STATS:
            vals = recent[stat].values
            if len(vals) < 3:
                continue
            trend = np.polyfit(range(len(vals)), vals, 1)[0]  # slope

            if abs(trend) < 1.0:   # less than 1 unit/poll — not significant
                continue

            direction = "rising" if trend > 0 else "falling"
            affected  = query_correlation_map(
                self.correlation_map, stat, direction
            )

            if not affected:
                continue

            top_affected = affected[:3]   # top 3 correlated stats

            alerts.append(CorrelationAlert(
                trigger_stat=stat,
                trigger_direction=direction,
                affected_stats=top_affected,
                lag_minutes=top_affected[0]["lag_minutes"] if top_affected else 0,
                confidence=top_affected[0]["confidence"] if top_affected else 0,
                interpretation=(
                    f"{stat} is {direction} (slope={trend:.2f}/poll). "
                    f"Expected downstream: {', '.join(d['affected_stat'] for d in top_affected)}"
                )
            ))
        return alerts

    def _build_alarm_predictions(self, pred_alarm: np.ndarray) -> List[Dict]:
        alarms = []
        for i, name in enumerate(ALARM_NAMES):
            prob = float(pred_alarm[i])
            if prob > 0.3:   # report if >30% probability
                alarms.append({
                    "alarm_msg":   name,
                    "probability": round(prob, 3),
                    "severity":    "critical" if prob > 0.85 else "high" if prob > 0.65 else "medium",
                    "horizon_min": self.horizon * POLL_INTERVAL
                })
        alarms.sort(key=lambda x: x["probability"], reverse=True)
        return alarms

    def _compute_health(
        self,
        active_breaches: List[str],
        stat_preds: List[StatPrediction],
        pred_alarm: np.ndarray
    ) -> Tuple[str, float]:

        # Deduct from 100
        score = 100.0
        score -= len(active_breaches) * 15          # -15 per active breach

        critical_preds = sum(1 for sp in stat_preds if sp.severity == "critical")
        high_preds     = sum(1 for sp in stat_preds if sp.severity == "high")
        score -= critical_preds * 10
        score -= high_preds     * 5
        score -= float(pred_alarm.max()) * 10

        score = max(0, min(100, score))

        if score >= 80:   health = "healthy"
        elif score >= 55: health = "degrading"
        elif score >= 30: health = "warning"
        else:             health = "critical"

        return health, round(score, 1)

    # ── Formatted output ──────────────────────

    def format_result(self, result: InferenceResult) -> str:
        lines = [
            f"\n{'─'*65}",
            f"  CORRELATION ENGINE INFERENCE REPORT",
            f"  Device: {result.device_name} | Interface: {result.interface_name}",
            f"  Time:   {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Health: {result.overall_health.upper()} ({result.overall_score}/100)",
            f"{'─'*65}",
        ]

        if result.active_breaches:
            lines.append(f"\n🚨 ACTIVE ALARMS ({len(result.active_breaches)}):")
            for b in result.active_breaches:
                lines.append(f"   • {b}")

        if result.predicted_alarms:
            lines.append(f"\n⚠  PREDICTED ALARMS (next {self.horizon * POLL_INTERVAL} min):")
            for a in result.predicted_alarms[:5]:
                lines.append(
                    f"   • [{a['severity'].upper():8}] {a['alarm_msg']:<40} "
                    f"P={a['probability']:.0%}"
                )

        lines.append(f"\n📊 STAT FORECASTS (next {self.horizon * POLL_INTERVAL} min):")
        lines.append(f"   {'Stat':<10} {'Now':>6} {'T+5':>6} {'T+10':>6} {'T+30':>6} "
                     f"{'MaxBreachP':>11} {'Severity':<10} {'Time2Breach'}")
        lines.append(f"   {'─'*85}")
        for sp in result.stat_predictions:
            t2b = f"{sp.time_to_breach_min} min" if sp.time_to_breach_min else "—"
            lines.append(
                f"   {sp.stat_name:<10} {sp.current_value:>6.1f} "
                f"{sp.predicted_values[0]:>6.1f} "
                f"{sp.predicted_values[1]:>6.1f} "
                f"{sp.predicted_values[-1]:>6.1f} "
                f"{sp.max_breach_prob:>11.1%} "
                f"{sp.severity:<10} {t2b}"
            )

        if result.correlation_alerts:
            lines.append(f"\n🔗 CORRELATION ALERTS:")
            for ca in result.correlation_alerts[:4]:
                lines.append(f"   • {ca.interpretation}")

        lines.append(f"{'─'*65}\n")
        return "\n".join(lines)

    def to_json(self, result: InferenceResult) -> Dict:
        return {
            "device_name":    result.device_name,
            "interface_name": result.interface_name,
            "timestamp":      result.timestamp.isoformat(),
            "overall_health": result.overall_health,
            "overall_score":  result.overall_score,
            "active_breaches":result.active_breaches,
            "predicted_alarms": result.predicted_alarms,
            "stat_predictions": [
                {
                    "stat":              sp.stat_name,
                    "current":           sp.current_value,
                    "predicted_values":  sp.predicted_values,
                    "breach_probability":sp.breach_probability,
                    "max_breach_prob":   sp.max_breach_prob,
                    "time_to_breach_min":sp.time_to_breach_min,
                    "alarm_msg":         sp.alarm_msg,
                    "severity":          sp.severity,
                    "confidence":        sp.confidence,
                }
                for sp in result.stat_predictions
            ],
            "correlation_alerts": [
                {
                    "trigger_stat":      ca.trigger_stat,
                    "trigger_direction": ca.trigger_direction,
                    "affected_stats":    ca.affected_stats[:3],
                    "lag_minutes":       ca.lag_minutes,
                    "confidence":        ca.confidence,
                    "interpretation":    ca.interpretation,
                }
                for ca in result.correlation_alerts
            ],
        }