"""
Forecasting Performance Monitoring & Drift Detection (Phase 9)
ForecastIQ Backend

Tracks:
- Historical MAPE and RMSE across chronological backtest folds and evaluation windows.
- Compares recent window accuracy against baseline history.
- Flags performance drift and emits actionable retraining review alerts.

Statuses:
- HEALTHY
- WATCH
- ACCURACY DRIFT
- INSUFFICIENT DATA
"""
from typing import Dict, Any, List, Optional
from ..data.schema import MonitoringReport
from ..evaluation.backtesting import BacktestingEngine

# In-memory monitoring cache: dataset_id -> MonitoringReport dict
MONITORING_CACHE: Dict[str, Dict[str, Any]] = {}

# Configurable drift thresholds
DRIFT_WATCH_THRESHOLD_PCT = 10.0
DRIFT_ALERT_THRESHOLD_PCT = 25.0


class MonitoringTracker:
    """
    Tracks model performance over time and detects accuracy degradation.
    """

    @classmethod
    def evaluate_monitoring_status(
        cls,
        dataset_id: str,
        backtest_summary: Optional[Dict[str, Any]] = None,
        watch_threshold: float = DRIFT_WATCH_THRESHOLD_PCT,
        alert_threshold: float = DRIFT_ALERT_THRESHOLD_PCT,
    ) -> MonitoringReport:
        """
        Evaluate chronological accuracy across backtesting folds to detect potential performance decay.
        """
        summary = backtest_summary or BacktestingEngine.get_evaluation_for_dataset(dataset_id)

        if not summary:
            return MonitoringReport(
                dataset_id=dataset_id,
                status="INSUFFICIENT DATA",
                retraining_alert=None,
                is_alert=False,
                models_tracked=[],
                details=[{"message": "No backtesting evaluations available yet for monitoring."}],
            )

        series_details = summary.get("series_details", [])
        models_tracked = summary.get("models_evaluated", [])

        # Collect chronological fold MAPEs across series
        fold_mapes: Dict[int, List[float]] = {}
        for s in series_details:
            folds = s.get("folds", [])
            for f in folds:
                f_num = f.get("fold", 1)
                f_mape = f.get("mape")
                if f_mape is not None and f.get("status") == "SUCCESS":
                    fold_mapes.setdefault(f_num, []).append(float(f_mape))

        if len(fold_mapes) < 2:
            # Need at least 2 distinct folds to compute drift between baseline and recent
            first_fold_values = fold_mapes.get(1, [])
            baseline_mape = round(float(sum(first_fold_values) / len(first_fold_values)), 2) if first_fold_values else None
            return MonitoringReport(
                dataset_id=dataset_id,
                status="INSUFFICIENT DATA" if len(fold_mapes) == 0 else "HEALTHY",
                baseline_mape=baseline_mape,
                recent_mape=baseline_mape,
                mape_drift_pct=0.0,
                retraining_alert=None,
                is_alert=False,
                models_tracked=models_tracked,
                details=[{"message": f"Only {len(fold_mapes)} fold(s) evaluated. More backtest folds required for multi-period drift detection."}],
            )

        sorted_fold_nums = sorted(fold_mapes.keys())
        first_fold_num = sorted_fold_nums[0]
        latest_fold_num = sorted_fold_nums[-1]

        base_vals = fold_mapes[first_fold_num]
        recent_vals = fold_mapes[latest_fold_num]

        baseline_mape = round(float(sum(base_vals) / len(base_vals)), 2) if base_vals else 0.0
        recent_mape = round(float(sum(recent_vals) / len(recent_vals)), 2) if recent_vals else 0.0

        if baseline_mape > 0:
            drift_pct = round(((recent_mape - baseline_mape) / baseline_mape) * 100.0, 2)
        else:
            drift_pct = 0.0

        if drift_pct >= alert_threshold:
            status = "ACCURACY DRIFT"
            retraining_alert = "Model performance has degraded beyond configured error thresholds. Review or retrain the model."
            is_alert = True
        elif drift_pct >= watch_threshold:
            status = "WATCH"
            retraining_alert = "Minor error growth observed in latest evaluation window. Continue monitoring upcoming actuals."
            is_alert = False
        else:
            status = "HEALTHY"
            retraining_alert = None
            is_alert = False

        details = [
            {
                "fold": first_fold_num,
                "label": "Baseline Window",
                "avg_mape": baseline_mape,
                "sample_evaluations": len(base_vals),
            },
            {
                "fold": latest_fold_num,
                "label": "Most Recent Window",
                "avg_mape": recent_mape,
                "sample_evaluations": len(recent_vals),
            }
        ]

        report = MonitoringReport(
            dataset_id=dataset_id,
            status=status,
            baseline_mape=baseline_mape,
            recent_mape=recent_mape,
            mape_drift_pct=drift_pct,
            retraining_alert=retraining_alert,
            is_alert=is_alert,
            models_tracked=models_tracked,
            details=details,
        )

        dumped = report.model_dump()
        MONITORING_CACHE[dataset_id] = dumped

        try:
            from ..db.mongo import MongoRepository
            MongoRepository.save_monitoring_report(dataset_id, dumped)
        except Exception:
            pass

        return report

    @classmethod
    def get_monitoring_report(cls, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached monitoring report if present in-memory or MongoDB."""
        if dataset_id in MONITORING_CACHE:
            return MONITORING_CACHE[dataset_id]

        try:
            from ..db.mongo import MongoRepository
            db_res = MongoRepository.get_monitoring_report(dataset_id)
            if db_res:
                MONITORING_CACHE[dataset_id] = db_res
                return db_res
        except Exception:
            pass

        return None
