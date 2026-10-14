# ForecastIQ — Multi-Horizon Time-Series Forecasting for Enterprise Analytics

ForecastIQ is a dataset-agnostic, production-ready enterprise forecasting and decision-support intelligence platform. Built with a FastAPI backend and a React/Tailwind/Vite frontend, ForecastIQ integrates classical statistical models, modern gradient boosted decision trees, and state-of-the-art deep learning sequence transformers to deliver robust, multi-horizon demand forecasts and actionable operational recommendations.

---

## Architecture Overview

```
Custom Enterprise CSV / M5 Benchmark
                ↓
    Dataset Profiler & Schema Validation
                ↓
    Universal Canonical Standard Schema
      (date, series_id, target + optional)
                ↓
    Auditable Data Cleaning & Quality Engine
                ↓
    Feature Classification & Engineering
   (Calendar, Lags, Rolling, Covariates)
                ↓
    Chronological Train / Validation / Test Splitting
                ↓
    Unified Multi-Horizon Forecasting Engine
  (ARIMA, SARIMA, Prophet, LightGBM, DeepAR, TFT)
                ↓
   Chronological Walk-Forward Backtesting (MAPE + RMSE)
                ↓
   Empirical Best-Model Selection Matrix
                ↓
   Business Decision-Support Engine & Monitoring
                ↓
   MongoDB Persistence Layer (with In-Memory Fallback)
```

---

## Universal Dataset Standard

Every uploaded time-series dataset must adhere to the universal canonical schema:

### Mandatory Columns
1. **`date`**: Timestamp / observation date (Daily frequency recommended).
2. **`series_id`**: Distinct time-series identifier (e.g., store, SKU, warehouse, branch, product).
3. **`target`**: Numerical quantity to forecast (e.g., demand, sales, revenue, orders).

### Supported Optional Columns
- `store_id`, `item_id`, `category`, `company`, `region`, `state`
- `price` (historical unit price)
- `promotion` (binary promotional flag)
- `holiday` (calendar holiday / event name)
- `weather`, `customer_segment`, etc.

*Note:* Exogenous features are validated to prevent future target leakage.

---

## Supported Forecasting Architectures (All 6 Models)

ForecastIQ supports six distinct models evaluated across **7-Day (Operational)**, **28-Day (Tactical)**, and **90-Day (Strategic)** planning horizons:

1. **ARIMA** (`statsmodels`): Classical autoregressive integrated moving average baseline with closed-form confidence bounds.
2. **SARIMA** (`statsmodels`): Seasonal statistical forecasting accounting for weekly recurring seasonality.
3. **Prophet** (`prophet`): Additive trend and seasonal decomposition with genuine Bayesian uncertainty intervals.
4. **LightGBM** (`lightgbm`): High-throughput gradient boosted trees utilizing chronological lag and rolling features.
5. **DeepAR** (`darts` + `PyTorch`): Probabilistic autoregressive recurrent neural network producing empirical prediction intervals.
6. **Temporal Fusion Transformer (TFT)** (`darts` + `PyTorch`): Multi-horizon self-attention architecture leveraging temporal covariates and external features.

---

## Evaluation & Decision Support

- **Chronological Walk-Forward Backtesting**: Rolling-origin evaluation across multi-fold historical windows ensuring strict temporal separation without future data leakage.
- **MAPE & RMSE**: Zero-actual protected MAPE metrics avoiding division-by-zero artifacts.
- **Empirical Best Model Selection**: Data-driven winner selection without hardcoded preferences.
- **Business Insights**: Actionable guidance for Inventory replenishment, Staffing shifts, and Procurement vendor lead times based strictly on computed forecast deltas.
- **Model Health Monitoring**: Error drift tracking (`HEALTHY`, `WATCH`, `ACCURACY DRIFT`) with automated advisory retraining alerts.

---

## MongoDB Persistence Layer

- Configured via `MONGODB_URI` and `MONGODB_DATABASE`.
- Persists datasets, metadata, forecast runs, backtest evaluations, business insights, and monitoring reports.
- Includes automatic fallback to in-memory caching when MongoDB is not running, ensuring uninterrupted offline development.

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- (Optional) MongoDB 6.0+

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Running Verification & Tests

### Backend Tests
```bash
pytest backend/tests -v
```
*Current Suite: 134 passed, 0 failures.*

### Frontend Tests, Lint, and Build
```bash
cd frontend
npm test -- --run   # 12 passed, 0 failures
npm run lint        # 0 errors, 0 warnings
npm run build       # Production build successful
```

---

## Known Genuine Limitations
- Real-time training of DeepAR and TFT over thousands of series simultaneously is computationally intensive; run backtesting on selected series slices or with GPU acceleration.
- Point forecasts generated by LightGBM and ARIMA without parametric assumptions do not emit interval bounds; uncertainty is appropriately flagged as `UNKNOWN` rather than fabricated.
