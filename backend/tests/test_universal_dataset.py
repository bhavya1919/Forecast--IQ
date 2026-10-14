"""
Comprehensive Backend Tests for Universal Dataset Engine (Phase 2)
Tests 1-14:
1. Valid canonical CSV
2. Missing date
3. Missing series_id
4. Missing target
5. Invalid date
6. Invalid target
7. Empty required field
8. Exact duplicate rows
9. Duplicate series_id + date
10. Date gaps
11. Frequency detection
12. Feature classification
13. M5 adapter
14. Dataset readiness
"""
import pytest
import pandas as pd
import numpy as np
from app.data.schema import (
    ReadinessStatus,
    ColumnClassification,
    MANDATORY_CANONICAL_FIELDS,
)
from app.data.loader import load_csv_from_bytes, detect_candidate_columns
from app.data.profiler import profile_dataset, detect_series_frequency
from app.data.validator import validate_schema_and_series
from app.features.analyzer import analyze_features
from app.data.adapters.m5_adapter import M5Adapter
from app.data.cleaner import evaluate_dataset_readiness


# 1. Valid canonical CSV
def test_valid_canonical_csv():
    csv_bytes = b"""date,series_id,target,store_id,price
2024-01-01,STORE_1_ITEM_A,42.0,STORE_1,2.50
2024-01-02,STORE_1_ITEM_A,45.0,STORE_1,2.50
2024-01-03,STORE_1_ITEM_A,50.0,STORE_1,2.50
2024-01-04,STORE_1_ITEM_A,48.0,STORE_1,2.50
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "valid_test.csv")
    mapping = detect_candidate_columns(list(df.columns))
    
    assert mapping["date"] == "date"
    assert mapping["series_id"] == "series_id"
    assert mapping["target"] == "target"
    
    val_report, summary = validate_schema_and_series(dataset_id, df, mapping)
    assert val_report.status == ReadinessStatus.READY
    assert len(val_report.errors) == 0
    assert summary.total_series == 1


# 2. Missing date
def test_missing_date():
    csv_bytes = b"""series_id,target,price
STORE_1_ITEM_A,42.0,2.50
STORE_1_ITEM_A,45.0,2.50
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "missing_date.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert "date" in val_report.missing_mandatory_fields
    assert any("date" in err.lower() for err in val_report.errors)
    assert val_report.status == ReadinessStatus.INVALID


# 3. Missing series_id
def test_missing_series_id():
    csv_bytes = b"""date,target,price
2024-01-01,42.0,2.50
2024-01-02,45.0,2.50
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "missing_series.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert "series_id" in val_report.missing_mandatory_fields
    assert any("series_id" in err.lower() for err in val_report.errors)
    assert val_report.status == ReadinessStatus.INVALID


# 4. Missing target
def test_missing_target():
    csv_bytes = b"""date,series_id,price
2024-01-01,STORE_1_ITEM_A,2.50
2024-01-02,STORE_1_ITEM_A,2.50
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "missing_target.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert "target" in val_report.missing_mandatory_fields
    assert any("target" in err.lower() for err in val_report.errors)
    assert val_report.status == ReadinessStatus.INVALID


# 5. Invalid date
def test_invalid_date():
    csv_bytes = b"""date,series_id,target
2024-01-01,STORE_1_ITEM_A,40.0
NOT_A_REAL_DATE,STORE_1_ITEM_A,45.0
2024-01-03,STORE_1_ITEM_A,50.0
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "invalid_date.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert val_report.invalid_dates_count == 1
    assert any("unparseable" in err.lower() or "malformed" in err.lower() for err in val_report.errors)
    assert val_report.status == ReadinessStatus.NEEDS_ATTENTION


# 6. Invalid target
def test_invalid_target():
    csv_bytes = b"""date,series_id,target
2024-01-01,STORE_1_ITEM_A,40.0
2024-01-02,STORE_1_ITEM_A,NOT_A_NUMBER
2024-01-03,STORE_1_ITEM_A,50.0
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "invalid_target.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert val_report.invalid_target_values_count == 1
    assert any("non-numeric" in err.lower() for err in val_report.errors)
    assert val_report.status == ReadinessStatus.NEEDS_ATTENTION


# 7. Empty required field
def test_empty_required_field():
    csv_bytes = b"""date,series_id,target
2024-01-01,,40.0
,STORE_1_ITEM_A,45.0
2024-01-03,STORE_1_ITEM_A,
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "empty_fields.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert any("empty or null series" in err.lower() for err in val_report.errors)
    assert any("empty or blank timestamp" in err.lower() for err in val_report.errors)


# 8. Exact duplicate rows
def test_exact_duplicate_rows():
    csv_bytes = b"""date,series_id,target
2024-01-01,STORE_1_ITEM_A,40.0
2024-01-01,STORE_1_ITEM_A,40.0
2024-01-02,STORE_1_ITEM_A,45.0
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "exact_dup.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert val_report.exact_duplicate_rows == 1
    assert any("exact duplicate" in warn.lower() for warn in val_report.warnings)


# 9. Duplicate series_id + date (conflict)
def test_duplicate_series_date_conflict():
    csv_bytes = b"""date,series_id,target
2024-01-01,STORE_1_ITEM_A,40.0
2024-01-01,STORE_1_ITEM_A,95.0
2024-01-02,STORE_1_ITEM_A,45.0
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "conflict_dup.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, _ = validate_schema_and_series(dataset_id, df, mapping)
    
    assert val_report.conflicting_series_date_records == 1
    assert any("conflicting record" in err.lower() for err in val_report.errors)


# 10. Date gaps
def test_date_gaps_detection():
    csv_bytes = b"""date,series_id,target
2024-01-01,STORE_1_ITEM_A,40.0
2024-01-02,STORE_1_ITEM_A,45.0
2024-01-08,STORE_1_ITEM_A,50.0
"""
    dataset_id, df = load_csv_from_bytes(csv_bytes, "date_gaps.csv")
    mapping = detect_candidate_columns(list(df.columns))
    val_report, series_summary = validate_schema_and_series(dataset_id, df, mapping)
    
    assert series_summary.series_with_gaps == 1
    assert any("temporal date gaps" in warn.lower() for warn in val_report.warnings)


# 11. Frequency detection
def test_frequency_detection():
    # Daily
    daily_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "series_id": "SERIES_1",
        "target": 10
    })
    freq_daily, _ = detect_series_frequency(daily_df, "date", "series_id")
    assert "Daily" in freq_daily

    # Weekly
    weekly_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="W"),
        "series_id": "SERIES_1",
        "target": 10
    })
    freq_weekly, warn_weekly = detect_series_frequency(weekly_df, "date", "series_id")
    assert "Weekly" in freq_weekly
    assert "Detected weekly frequency" in warn_weekly


# 12. Feature classification
def test_feature_classification():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "series_id": ["S1", "S1", "S1"],
        "target": [10.0, 15.0, 20.0],
        "holiday_event": ["Easter", "", ""],
        "future_demand_test": [12.0, 14.0, 16.0],
        "constant_col": [1, 1, 1],
        "store_id": ["STORE_A", "STORE_A", "STORE_A"],
        "price": [2.99, 2.99, 1.99]
    })
    mapping = {
        "date": "date",
        "series_id": "series_id",
        "target": "target"
    }
    features = analyze_features(df, mapping)
    class_map = {f.name: f.classification for f in features}

    assert class_map["date"] == ColumnClassification.TIME
    assert class_map["target"] == ColumnClassification.TARGET
    assert class_map["series_id"] == ColumnClassification.SERIES_ID
    assert class_map["holiday_event"] == ColumnClassification.KNOWN_FUTURE
    assert class_map["future_demand_test"] == ColumnClassification.POTENTIAL_LEAKAGE
    assert class_map["constant_col"] == ColumnClassification.CONSTANT
    assert class_map["price"] == ColumnClassification.POTENTIALLY_USEFUL


# 13. M5 adapter
def test_m5_adapter():
    m5_sales = pd.DataFrame({
        "id": ["HOBBIES_1_001_CA_1_validation", "HOBBIES_1_002_CA_1_validation"],
        "item_id": ["HOBBIES_1_001", "HOBBIES_1_002"],
        "dept_id": ["HOBBIES_1", "HOBBIES_1"],
        "cat_id": ["HOBBIES", "HOBBIES"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
        "d_1": [0, 1],
        "d_2": [2, 0],
        "d_3": [1, 3]
    })
    calendar_df = pd.DataFrame({
        "d": ["d_1", "d_2", "d_3"],
        "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
        "wm_yr_wk": [11101, 11101, 11101],
        "event_name_1": ["NewYear", None, None]
    })

    canonical_df = M5Adapter.transform_sales_wide_to_canonical(
        sales_df=m5_sales,
        calendar_df=calendar_df,
        max_series=2,
        max_days=3
    )

    assert "date" in canonical_df.columns
    assert "series_id" in canonical_df.columns
    assert "target" in canonical_df.columns
    assert "store_id" in canonical_df.columns
    assert "category" in canonical_df.columns
    assert len(canonical_df) == 6  # 2 series * 3 days
    assert canonical_df["series_id"].iloc[0] == "HOBBIES_1_001_CA_1"


# 14. Dataset readiness
def test_dataset_readiness_evaluation():
    # Construct 45-day daily series
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "series_id": "SERIES_CA_1",
        "target": np.random.randint(10, 50, size=60),
        "price": 2.50
    })
    mapping = {"date": "date", "series_id": "series_id", "target": "target"}
    report = evaluate_dataset_readiness("test-id-123", df, "retail_60d.csv", mapping)

    assert report.status == ReadinessStatus.READY
    assert report.profile.rows_count == 60
    assert report.profile.unique_series_count == 1
    assert report.supported_horizons["h7"].status == "Supported"
    assert report.supported_horizons["h28"].status == "Borderline" or report.supported_horizons["h28"].status == "Supported"
    assert len(report.readiness_checklist) == 5
    assert all(item["passed"] for item in report.readiness_checklist)
