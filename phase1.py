"""
WEC Performance Analysis - Phase 1
===================================
Data-driven regression model to predict wave energy converter output
(Energy_Generation_kW) from meteoceanographic sensor data.

Model choice: XGBoost Regressor (ensemble tree method).
Rationale: strong performance on tabular data, native handling of missing
values, built-in feature importance, and full explainability to management.
The architecture allows straightforward substitution with RandomForest or
LightGBM via the MODEL_REGISTRY dictionary in `train_model`.

Pipeline:
    1. load_and_preprocess_data  -- ingest, clean, engineer features
    2. prepare_features          -- select features, temporal train/test split
    3. train_model               -- fit and evaluate XGBoost
    4. save_model / load_model   -- persist and reload for later inference
    5. predict_new_data          -- run inference on new data batches
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target variable name
TARGET_COL = "Energy_Generation_kW"

# Timestamp column used for temporal ordering and feature extraction
TIMESTAMP_COL = "PCTimeStamp"

# Feature set selected for the model based on Waverider API data.
FEATURE_COLS = [
    "Hs__m",
    "Te__s",
    "Wave_Power_Flux",       # engineered feature (0.49 * Hs^2 * Te)
    "H1/3__m",
    "H1/10__m",
    "Hmax__m",
    "HTmax__m",
    "Havg__m",
    "Hsms__m",
    "NumberOfWaves",
    "THmax__s",
    "Tavg__s",
    "Tmax__s",
    # Temporal features
    "hour",
    "month",
]

# Fraction of data reserved for testing (held out as the most recent portion)
TEST_FRACTION = 0.20

# Default path for saving the trained model artifact
DEFAULT_MODEL_PATH = "wec_phase1_model.joblib"

# ---------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess_data(
    source: Union[str, Path, pd.DataFrame],
    remove_outliers: bool = True,
    outlier_z_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Ingest raw sensor data, clean it, and engineer new features.
    """

    # ------------------------------------------------------------------
    # Step 1a -- Ingest
    # ------------------------------------------------------------------
    if isinstance(source, (str, Path)):
        logger.info("Loading data from: %s", source)
        df = pd.read_csv(source, parse_dates=[TIMESTAMP_COL])
    elif isinstance(source, pd.DataFrame):
        logger.info("Received in-memory DataFrame with %d rows", len(source))
        df = source.copy()
        if TIMESTAMP_COL in df.columns and not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    else:
        raise TypeError("source must be a file path or a pandas DataFrame")

    logger.info("Raw shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Step 1b -- Sort by timestamp to guarantee temporal ordering
    # ------------------------------------------------------------------
    if TIMESTAMP_COL in df.columns:
        df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 1c & 1d -- Outlier handling (flag-and-impute strategy)
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if remove_outliers:
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_std  = df[col].std()
            if col_std == 0:
                continue
            z_scores = (df[col] - col_mean).abs() / col_std
            n_outliers = (z_scores > outlier_z_threshold).sum()
            if n_outliers > 0:
                logger.debug("Column '%s': flagging %d outlier(s) as NaN", col, n_outliers)
                df.loc[z_scores > outlier_z_threshold, col] = np.nan

    # ------------------------------------------------------------------
    # Step 1e -- Imputation of missing values
    # ------------------------------------------------------------------
    missing_before = df[numeric_cols].isna().sum().sum()
    if missing_before > 0:
        logger.info("Imputing %d missing value(s) across numeric columns", missing_before)
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # ------------------------------------------------------------------
    # Step 1f -- Feature Engineering: Physics-based variables
    # ------------------------------------------------------------------
    if "Hs__m" in df.columns and "Te__s" in df.columns:
        # Physical Wave Power Flux (kW/m)
        df["Wave_Power_Flux"] = 0.49 * (df["Hs__m"] ** 2) * df["Te__s"]
        logger.info("Feature 'Wave_Power_Flux' computed using Hs and Te")

    # ------------------------------------------------------------------
    # Step 1g -- Temporal feature extraction
    # ------------------------------------------------------------------
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        df["hour"]  = df[TIMESTAMP_COL].dt.hour
        df["month"] = df[TIMESTAMP_COL].dt.month
        logger.info("Temporal features extracted: hour, month")

    logger.info("Preprocessed shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. Feature Selection and Train/Test Split
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    test_fraction: float = TEST_FRACTION,
    is_inference: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame, Optional[pd.Series]]:
    """
    Select the model features and produce a temporally-correct split.
    """

    cols = feature_cols or FEATURE_COLS

    available_cols = [c for c in cols if c in df.columns]
    missing_requested = set(cols) - set(available_cols)
    if missing_requested:
        logger.warning("Requested feature(s) not found and will be skipped: %s", missing_requested)

    X = df[available_cols].copy()

    if is_inference:
        logger.info("Inference mode -- returning all %d rows as X", len(X))
        return X, None, None, None

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found.  "
            "Use is_inference=True if target is unavailable."
        )

    y = df[TARGET_COL].copy()

    split_idx = int(len(df) * (1 - test_fraction))
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    logger.info(
        "Temporal split: train=%d rows | test=%d rows (%.0f%% / %.0f%%)",
        len(X_train), len(X_test),
        (1 - test_fraction) * 100, test_fraction * 100,
    )

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Any] = {
    "xgboost": XGBRegressor,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    42,
    "n_jobs":         -1,
    "verbosity":       0,
}

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    model_type: str = "xgboost",
    model_params: Optional[dict] = None,
) -> Any:
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_type]
    params = model_params or (DEFAULT_XGB_PARAMS if model_type == "xgboost" else {})
    model  = ModelClass(**params)

    logger.info("Training %s on %d samples with %d features...", model_type, len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    logger.info("--- Test Set Metrics ---")
    logger.info("  RMSE : %.4f kW", rmse)
    logger.info("  MAE  : %.4f kW", mae)
    logger.info("  R^2  : %.4f",    r2)
    logger.info("------------------------")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)
        logger.info("Top feature importances:\n%s", importances.to_string())

    return model


# ---------------------------------------------------------------------------
# 4. Model Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    logger.info("Model saved to: %s", path.resolve())

def load_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file found at: {path.resolve()}")
    model = joblib.load(path)
    logger.info("Model loaded from: %s", path.resolve())
    return model


# ---------------------------------------------------------------------------
# 5. Inference on New Data
# ---------------------------------------------------------------------------

def predict_new_data(
    model: Any,
    source: Union[str, Path, pd.DataFrame],
    feature_cols: Optional[list] = None,
) -> pd.Series:
    
    df_new = load_and_preprocess_data(source)
    X_new, _, _, _ = prepare_features(
        df_new,
        feature_cols=feature_cols,
        is_inference=True,
    )

    logger.info("Running inference on %d samples...", len(X_new))
    predictions = model.predict(X_new)
    pred_series = pd.Series(predictions, index=X_new.index, name="Predicted_Energy_kW")

    n_out_of_bounds = ((pred_series < 0) | (pred_series > 350)).sum()
    if n_out_of_bounds > 0:
        logger.warning("Clipping %d prediction(s) to physical bounds [0, 350] kW", n_out_of_bounds)
        pred_series = pred_series.clip(lower=0, upper=350)

    logger.info(
        "Inference complete. Predicted range: [%.2f, %.2f] kW",
        pred_series.min(), pred_series.max(),
    )

    return pred_series


# ---------------------------------------------------------------------------
# Utility: generate a synthetic dataset for demonstration
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq="30min")
    
    # Base physics simulation for the waverider variables
    hs = rng.uniform(0.5, 5.0, n_rows)
    te = rng.uniform(4.0, 14.0, n_rows)
    
    wave_power_flux = 0.49 * (hs ** 2) * te
    capture_width_ratio = 2.5
    energy = (wave_power_flux * capture_width_ratio).clip(min=0, max=350)

    df = pd.DataFrame({
        TIMESTAMP_COL: timestamps,
        "Buoy_ID": "Boia_1",
        "Hs__m": hs,
        "Te__s": te,
        "H1/3__m": hs * 1.05 + rng.normal(0, 0.05, n_rows),
        "H1/10__m": hs * 1.27 + rng.normal(0, 0.05, n_rows),
        "Hmax__m": hs * 1.7 + rng.normal(0, 0.1, n_rows),
        "HTmax__m": hs * 1.5 + rng.normal(0, 0.1, n_rows),
        "Havg__m": hs * 0.6 + rng.normal(0, 0.05, n_rows),
        "Hsms__m": hs * 1.1 + rng.normal(0, 0.05, n_rows),
        "NumberOfWaves": rng.integers(100, 300, n_rows),
        "THmax__s": te * 1.2 + rng.normal(0, 0.5, n_rows),
        "Tavg__s": te * 0.8 + rng.normal(0, 0.2, n_rows),
        "Tmax__s": te * 1.5 + rng.normal(0, 0.5, n_rows),
        TARGET_COL: energy,
        "Epoch_Marker": range(1, n_rows + 1),
    })

    # Randomly insert ~2% missing values to validate imputation logic
    for col in ["Hs__m", "Te__s"]:
        mask = rng.random(n_rows) < 0.02
        df.loc[mask, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Main -- Full pipeline demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("WEC Phase 1 -- Full Pipeline Demo")
    logger.info("=" * 60)

    DATA_CSV_PATH = "dataset2/wec_c5_mock_data_epochs.csv"
    MODEL_SAVE_PATH = "wec_phase1_model.joblib"

    logger.info("[PHASE A] Training on historical data")

    if DATA_CSV_PATH and os.path.exists(DATA_CSV_PATH):
        logger.info("Using real data file: %s", DATA_CSV_PATH)
        df_historical = load_and_preprocess_data(DATA_CSV_PATH)
    else:
        logger.info("CSV not found -- generating synthetic dataset for demo")
        raw_df = _generate_synthetic_data(n_rows=2000)
        df_historical = load_and_preprocess_data(raw_df)

    X_train, y_train, X_test, y_test = prepare_features(df_historical)

    model = train_model(X_train, y_train, X_test, y_test, model_type="xgboost")

    save_model(model, path=MODEL_SAVE_PATH)

    logger.info("[PHASE B] Inference on a new incoming data batch")

    loaded_model = load_model(path=MODEL_SAVE_PATH)

    new_batch_raw = _generate_synthetic_data(n_rows=50, seed=99)
    new_batch_no_target = new_batch_raw.drop(columns=[TARGET_COL])

    predictions = predict_new_data(loaded_model, source=new_batch_no_target)

    sample = pd.DataFrame({
        TIMESTAMP_COL: new_batch_raw[TIMESTAMP_COL].values[:10],
        "Hs__m":       new_batch_raw["Hs__m"].values[:10],
        "Te__s":       new_batch_raw["Te__s"].values[:10],
        "Predicted_Energy_kW": predictions.values[:10],
    })

    logger.info("Sample predictions (first 10 rows):\n%s", sample.to_string(index=False))

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 60)