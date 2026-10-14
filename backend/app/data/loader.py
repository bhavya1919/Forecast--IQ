"""
Safe Data Loader and In-Memory Storage
ForecastIQ Backend
"""
import io
import uuid
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from .schema import MANDATORY_CANONICAL_FIELDS, OPTIONAL_CANONICAL_FIELDS

# In-memory session store (no MongoDB required in Phase 2)
DATASET_STORE: Dict[str, Dict[str, Any]] = {}


def detect_candidate_columns(columns: List[str]) -> Dict[str, Any]:
    """
    Intelligently map user columns to canonical schema fields using
    case-insensitive matching and standard alias heuristics.
    """
    col_lower = {col.lower().strip(): col for col in columns}

    def match_alias(aliases: List[str]) -> Optional[str]:
        for alias in aliases:
            if alias.lower() in col_lower:
                return col_lower[alias.lower()]
        return None

    # Mandatory
    date_col = match_alias(["date", "datetime", "timestamp", "ds", "time", "d", "observation_date"])
    series_col = match_alias(["series_id", "series", "item_id", "id", "store_item", "unique_id", "ts_id", "sku", "sku_code", "sku_id", "product_code"])
    target_col = match_alias(["target", "sales", "demand", "units", "y", "quantity", "val", "value", "revenue"])

    # Optional canonical
    optional_mapped: Dict[str, str] = {}
    for opt_field in OPTIONAL_CANONICAL_FIELDS:
        if opt_field == "store_id":
            matched = match_alias(["store_id", "store", "branch", "warehouse", "location"])
        elif opt_field == "item_id":
            matched = match_alias(["item_id", "item", "product_id", "product", "sku_id", "sku"])
        elif opt_field == "category":
            matched = match_alias(["category", "dept", "department", "cat", "product_category"])
        elif opt_field == "company":
            matched = match_alias(["company", "org", "enterprise", "business_unit"])
        elif opt_field == "region":
            matched = match_alias(["region", "district", "zone", "area", "market"])
        elif opt_field == "state":
            matched = match_alias(["state", "province", "territory"])
        elif opt_field == "price":
            matched = match_alias(["price", "sell_price", "unit_price", "selling_price", "base_price"])
        elif opt_field == "promotion":
            matched = match_alias(["promotion", "promo", "discount", "is_promo", "event", "deal"])
        elif opt_field == "holiday":
            matched = match_alias(["holiday", "event_name", "is_holiday", "holiday_name"])
        elif opt_field == "weather":
            matched = match_alias(["weather", "temperature", "temp", "precipitation", "rain"])
        elif opt_field == "customer_segment":
            matched = match_alias(["customer_segment", "segment", "channel", "customer_type"])
        else:
            matched = None

        if matched and matched != date_col and matched != series_col and matched != target_col:
            optional_mapped[opt_field] = matched

    # Find other unassigned columns
    all_assigned = {date_col, series_col, target_col, *optional_mapped.values()}
    for col in columns:
        if col not in all_assigned:
            optional_mapped[f"feature_{col}"] = col

    return {
        "date": date_col or "",
        "series_id": series_col or "",
        "target": target_col or "",
        "optional": optional_mapped,
    }


def load_csv_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, pd.DataFrame]:
    """
    Safely parse CSV from in-memory bytes, trying UTF-8 and fallback encodings.
    """
    dataset_id = str(uuid.uuid4())
    
    # Try UTF-8 first, fallback to latin-1
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")

    if df.empty and len(df.columns) == 0:
        raise ValueError("Uploaded CSV is completely empty.")

    # Clean whitespace in column headers
    df.columns = [str(c).strip() for c in df.columns]

    save_dataset(dataset_id, df, filename, metadata={"original_columns": list(df.columns)})
    return dataset_id, df


def save_dataset(dataset_id: str, df: pd.DataFrame, filename: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Store dataset in in-memory session cache and persist metadata to MongoDB."""
    meta = metadata or {}
    meta["rows"] = len(df)
    meta["columns"] = list(df.columns)
    meta["filename"] = filename

    DATASET_STORE[dataset_id] = {
        "df": df,
        "filename": filename,
        "metadata": meta,
    }

    try:
        from ..db.mongo import MongoRepository
        MongoRepository.save_dataset_metadata(dataset_id, meta)
    except Exception:
        pass


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored dataset from in-memory cache."""
    return DATASET_STORE.get(dataset_id)
