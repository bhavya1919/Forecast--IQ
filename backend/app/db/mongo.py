"""
ForecastIQ MongoDB Persistence & Repository Layer

Provides clean repository pattern implementations for saving and retrieving:
1. Datasets & metadata
2. Forecast runs & Forecast results
3. Evaluation & Backtesting comparisons
4. Business insights
5. Model monitoring reports

Gracefully handles environments where MongoDB is unreachable or unconfigured,
falling back without disrupting in-memory or file-based operations.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pymongo
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, PyMongoError

from ..config.settings import settings

logger = logging.getLogger(__name__)


class MongoManager:
    """Singleton connection manager for MongoDB with health-check and index management."""
    _client: Optional[MongoClient] = None
    _db = None
    _connected: bool = False
    _initialized_indexes: bool = False

    @classmethod
    def get_db(cls):
        """Returns the database instance if connected, else None."""
        if not settings.MONGODB_ENABLED:
            return None

        if cls._client is None:
            try:
                # Fast timeout (2s) so local development without mongo doesn't hang
                cls._client = MongoClient(
                    settings.MONGODB_URI,
                    serverSelectionTimeoutMS=2000,
                    connectTimeoutMS=2000
                )
                # Check connection
                cls._client.admin.command('ping')
                cls._db = cls._client[settings.MONGODB_DATABASE]
                cls._connected = True
                logger.info(f"Connected successfully to MongoDB database '{settings.MONGODB_DATABASE}'")
                cls.ensure_indexes()
            except Exception as e:
                cls._connected = False
                cls._db = None
                cls._client = None
                logger.debug(f"MongoDB not available ({e}). Running in memory/file mode.")

        return cls._db

    @classmethod
    def is_connected(cls) -> bool:
        """Check if active connection exists."""
        db = cls.get_db()
        return db is not None and cls._connected

    @classmethod
    def ensure_indexes(cls):
        """Build required indexes according to specification."""
        if cls._db is None or cls._initialized_indexes:
            return

        try:
            # 1. datasets collection: dataset_id unique
            cls._db.datasets.create_index([("dataset_id", ASCENDING)], unique=True)

            # 2. forecast_runs: dataset_id + series_id + model + horizon
            cls._db.forecast_runs.create_index([
                ("dataset_id", ASCENDING),
                ("series_id", ASCENDING),
                ("model", ASCENDING),
                ("forecast_horizon", ASCENDING)
            ])

            # 3. forecast_results: dataset_id + series_id + model + horizon
            cls._db.forecast_results.create_index([
                ("dataset_id", ASCENDING),
                ("series_id", ASCENDING),
                ("model", ASCENDING),
                ("forecast_horizon", ASCENDING)
            ], unique=True)

            # 4. evaluations: dataset_id + series_id + model + horizon
            cls._db.evaluations.create_index([
                ("dataset_id", ASCENDING),
                ("series_id", ASCENDING),
                ("model", ASCENDING),
                ("horizon", ASCENDING)
            ])
            # dataset_id index (non-unique: multiple scopes per dataset allowed)
            cls._db.evaluations.create_index([("dataset_id", ASCENDING)])

            # 5. business_insights: dataset_id + series_id + horizon
            cls._db.business_insights.create_index([
                ("dataset_id", ASCENDING),
                ("series_id", ASCENDING),
                ("horizon", ASCENDING),
                ("model", ASCENDING)
            ], unique=True)

            # 6. monitoring_reports: dataset_id + model
            cls._db.monitoring_reports.create_index([
                ("dataset_id", ASCENDING)
            ], unique=True)

            cls._initialized_indexes = True
            logger.info("MongoDB indexes successfully created and verified.")
        except Exception as e:
            logger.warning(f"Could not build MongoDB indexes: {e}")


class MongoRepository:
    """Repository handling document storage and retrieval."""

    @staticmethod
    def save_dataset_metadata(dataset_id: str, metadata: Dict[str, Any]) -> bool:
        db = MongoManager.get_db()
        if db is None:
            return False
        try:
            doc = {
                "dataset_id": dataset_id,
                "metadata": metadata,
                "updated_at": datetime.utcnow()
            }
            db.datasets.update_one(
                {"dataset_id": dataset_id},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            logger.warning(f"Error saving dataset metadata to MongoDB: {e}")
            return False

    @staticmethod
    def get_dataset_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
        db = MongoManager.get_db()
        if db is None:
            return None
        try:
            res = db.datasets.find_one({"dataset_id": dataset_id})
            if res and "metadata" in res:
                return res["metadata"]
            return None
        except Exception as e:
            logger.warning(f"Error fetching dataset metadata from MongoDB: {e}")
            return None

    @staticmethod
    def save_forecast_result(forecast_result_dict: Dict[str, Any]) -> bool:
        db = MongoManager.get_db()
        if db is None:
            return False
        try:
            query = {
                "dataset_id": forecast_result_dict.get("dataset_id"),
                "series_id": forecast_result_dict.get("series_id"),
                "model": forecast_result_dict.get("model"),
                "forecast_horizon": forecast_result_dict.get("forecast_horizon")
            }
            doc = {
                **forecast_result_dict,
                "updated_at": datetime.utcnow()
            }
            db.forecast_results.update_one(query, {"$set": doc}, upsert=True)
            return True
        except Exception as e:
            logger.warning(f"Error saving forecast result to MongoDB: {e}")
            return False

    @staticmethod
    def get_forecast_result(dataset_id: str, series_id: str, model: str, horizon: int) -> Optional[Dict[str, Any]]:
        db = MongoManager.get_db()
        if db is None:
            return None
        try:
            res = db.forecast_results.find_one({
                "dataset_id": dataset_id,
                "series_id": series_id,
                "model": model,
                "forecast_horizon": horizon
            }, {"_id": 0, "updated_at": 0})
            return res
        except Exception as e:
            logger.warning(f"Error getting forecast result from MongoDB: {e}")
            return None

    @staticmethod
    def save_evaluation_comparison(dataset_id: str, evaluation_dict: Dict[str, Any], scope_key: Optional[str] = None) -> bool:
        db = MongoManager.get_db()
        if db is None:
            return False
        try:
            sk = scope_key or evaluation_dict.get("scope_key") or "default"
            doc = {
                "dataset_id": dataset_id,
                "scope_key": sk,
                **evaluation_dict,
                "updated_at": datetime.utcnow()
            }
            db.evaluations.update_one(
                {"dataset_id": dataset_id, "scope_key": sk},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            logger.warning(f"Error saving evaluation comparison to MongoDB: {e}")
            return False

    @staticmethod
    def get_evaluation_comparison(dataset_id: str, scope_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        db = MongoManager.get_db()
        if db is None:
            return None
        try:
            q = {"dataset_id": dataset_id}
            if scope_key:
                q["scope_key"] = scope_key
            res = db.evaluations.find_one(q, {"_id": 0, "updated_at": 0})
            return res
        except Exception as e:
            logger.warning(f"Error fetching evaluation comparison from MongoDB: {e}")
            return None

    @staticmethod
    def save_business_insights(insights_dict: Dict[str, Any]) -> bool:
        db = MongoManager.get_db()
        if db is None:
            return False
        try:
            query = {
                "dataset_id": insights_dict.get("dataset_id"),
                "series_id": insights_dict.get("series_id"),
                "horizon": insights_dict.get("horizon"),
                "model": insights_dict.get("model")
            }
            doc = {
                **insights_dict,
                "updated_at": datetime.utcnow()
            }
            db.business_insights.update_one(query, {"$set": doc}, upsert=True)
            return True
        except Exception as e:
            logger.warning(f"Error saving business insights to MongoDB: {e}")
            return False

    @staticmethod
    def get_business_insights(dataset_id: str, series_id: str, horizon: int, model: str) -> Optional[Dict[str, Any]]:
        db = MongoManager.get_db()
        if db is None:
            return None
        try:
            res = db.business_insights.find_one({
                "dataset_id": dataset_id,
                "series_id": series_id,
                "horizon": horizon,
                "model": model
            }, {"_id": 0, "updated_at": 0})
            return res
        except Exception as e:
            logger.warning(f"Error getting business insights from MongoDB: {e}")
            return None

    @staticmethod
    def save_monitoring_report(dataset_id: str, report_dict: Dict[str, Any]) -> bool:
        db = MongoManager.get_db()
        if db is None:
            return False
        try:
            doc = {
                "dataset_id": dataset_id,
                **report_dict,
                "updated_at": datetime.utcnow()
            }
            db.monitoring_reports.update_one({"dataset_id": dataset_id}, {"$set": doc}, upsert=True)
            return True
        except Exception as e:
            logger.warning(f"Error saving monitoring report to MongoDB: {e}")
            return False

    @staticmethod
    def get_monitoring_report(dataset_id: str) -> Optional[Dict[str, Any]]:
        db = MongoManager.get_db()
        if db is None:
            return None
        try:
            res = db.monitoring_reports.find_one({"dataset_id": dataset_id}, {"_id": 0, "updated_at": 0})
            return res
        except Exception as e:
            logger.warning(f"Error getting monitoring report from MongoDB: {e}")
            return None
