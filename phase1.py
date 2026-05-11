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

# Feature set selected for the model.
# Physical rationale:
#   - Wave_Hs (significant wave height): primary driver of available wave power
#   - Wave_Tp (peak period): determines wave energy period
#   - Wave_Power_Flux: derived feature -- actual physical power density proxy
#   - Wave_Steepness: derived feature -- wave sharpness (impacts efficiency/loads)
#   - Misalignment: derived feature -- angle difference between wind and waves
#   - Wind_Power_Density: derived feature -- raw wind power potential
#   - Wind_Speed, Current_Speed, Wave_Dir: directly measured environmental drivers
FEATURE_COLS = [
    "Wave_Hs",
    "Wave_Tp",
    "Wave_Power_Flux",       # engineered feature
    "Wave_Steepness",        # engineered feature
    "Misalignment",          # engineered feature
    "Wind_Power_Density",    # engineered feature
    "Wind_Speed",
    "Current_Speed",
    "Wave_Dir",
    "Air_Temperature",
    "Atmospheric_Pressure",
    # Temporal features
    "hour",
    "month",
]

# Fraction of data reserved for testing (held out as the most recent portion)
TEST_FRACTION = 0.20

# Default path for saving the trained model artifact
DEFAULT_MODEL_PATH = "wec_phase1_model.joblib"

# Physical constant for wave power density formula (rho * g^2 / (64 * pi))
# Units: W / (m^3 * s) -- absorbed into a proportionality constant here
# We use a normalised version: Wave_Power_Flux = Hs^2 * Tp
# The true formula (deep water) is: P = (rho * g^2) / (64 * pi) * Hs^2 * Te
# For a relative feature the constant cancels in importance calculations,
# so we omit it and keep the physically meaningful product.
WAVE_POWER_SCALE = 1.0  # change to 1025*9.81**2/(64*3.14159) for SI watts/m


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
    if "Wave_Hs" in df.columns and "Wave_Tp" in df.columns:
        # Power Flux: Proxy for wave power density
        df["Wave_Power_Flux"] = WAVE_POWER_SCALE * df["Wave_Hs"] ** 2 * df["Wave_Tp"]
        # Wave Steepness: H / L proxy (using Tp^2 as proxy for wavelength)
        df["Wave_Steepness"] = df["Wave_Hs"] / (df["Wave_Tp"] ** 2)
        logger.info("Features 'Wave_Power_Flux' and 'Wave_Steepness' computed")
        
    if "Wave_Dir" in df.columns and "Wind_Direction" in df.columns:
        # Misalignment: Shortest angular difference between wind and waves (0 to 180 deg)
        diff = np.abs(df["Wave_Dir"] - df["Wind_Direction"])
        df["Misalignment"] = np.minimum(diff, 360 - diff)
        logger.info("Feature 'Misalignment' computed")
        
    if "Wind_Speed" in df.columns:
        # Wind Power Density: Proportional to velocity cubed
        df["Wind_Power_Density"] = df["Wind_Speed"] ** 3
        logger.info("Feature 'Wind_Power_Density' computed")

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

    Using a temporal split (not random shuffle) is mandatory for time-series
    data.  A random split would leak future information into the training set,
    producing optimistically biased evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from `load_and_preprocess_data`.
    feature_cols : list, optional
        Override the default FEATURE_COLS constant.
    test_fraction : float
        Fraction of rows (from the end of the time series) to use as the
        test set.  Default: 0.20 (most recent 20 percent of data).
    is_inference : bool
        If True, the target column is not expected in `df`.
        Returns (X_all, None, None, None) for inference mode.

    Returns
    -------
    X_train, y_train, X_test, y_test
        In inference mode: (X_all, None, None, None)
    """

    cols = feature_cols or FEATURE_COLS

    # Keep only columns that actually exist in this batch
    available_cols = [c for c in cols if c in df.columns]
    missing_requested = set(cols) - set(available_cols)
    if missing_requested:
        logger.warning("Requested feature(s) not found and will be skipped: %s", missing_requested)

    X = df[available_cols].copy()

    # ------------------------------------------------------------------
    # Inference mode: no target column expected
    # ------------------------------------------------------------------
    if is_inference:
        logger.info("Inference mode -- returning all %d rows as X", len(X))
        return X, None, None, None

    # ------------------------------------------------------------------
    # Training mode: extract target and perform temporal split
    # ------------------------------------------------------------------
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

# Registry of supported ensemble tree regressors.
# To switch from XGBoost to RandomForest or LightGBM, change MODEL_TYPE
# in the call to `train_model`.  All three share the sklearn fit/predict API.
MODEL_REGISTRY: Dict[str, Any] = {
    "xgboost": XGBRegressor,
    # Uncomment to enable alternatives:
    # "random_forest": RandomForestRegressor,  # from sklearn.ensemble
    # "lightgbm":      LGBMRegressor,          # from lightgbm
}

DEFAULT_XGB_PARAMS = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "reg_alpha":       0.1,   # L1 regularisation (reduces overfitting on correlated wave features)
    "reg_lambda":      1.0,   # L2 regularisation
    "random_state":    42,
    "n_jobs":         -1,
    "verbosity":       0,
    # "objective": "reg:quantileerror",
    # "quantile_alpha": [0.1, 0.5, 0.9],
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    model_type: str = "xgboost",
    model_params: Optional[dict] = None,
) -> Any:
    """
    Fit an ensemble tree regressor and report evaluation metrics.

    Parameters
    ----------
    X_train, y_train : training features and target
    X_test,  y_test  : held-out test features and target
    model_type : str
        Key into MODEL_REGISTRY.  Default: 'xgboost'.
    model_params : dict, optional
        Hyperparameters forwarded to the model constructor.
        Falls back to DEFAULT_XGB_PARAMS for XGBoost.

    Returns
    -------
    Fitted model object (sklearn-compatible API).
    """

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_type]

    # Use provided params or fall back to sensible defaults
    params = model_params or (DEFAULT_XGB_PARAMS if model_type == "xgboost" else {})
    model  = ModelClass(**params)

    logger.info("Training %s on %d samples with %d features...", model_type, len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Evaluation on the held-out test set
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    logger.info("--- Test Set Metrics ---")
    logger.info("  RMSE : %.4f kW", rmse)
    logger.info("  MAE  : %.4f kW", mae)
    logger.info("  R^2  : %.4f",    r2)
    logger.info("------------------------")

    # ------------------------------------------------------------------
    # Feature importance (gain-based, XGBoost native)
    # ------------------------------------------------------------------
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
    """
    Serialise a fitted model to disk using joblib.

    joblib is preferred over pickle for scikit-learn / XGBoost objects
    because it handles large numpy arrays more efficiently via memory-mapped
    files, and produces smaller files through optional compression.

    Parameters
    ----------
    model : fitted model object
    path  : destination file path (e.g. 'wec_phase1_model.joblib')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    logger.info("Model saved to: %s", path.resolve())


def load_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Any:
    """
    Deserialise a previously saved model from disk.

    Parameters
    ----------
    path : path to the .joblib file produced by `save_model`

    Returns
    -------
    Fitted model object ready for inference.
    """
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
    """
    Run the full preprocessing pipeline on a new data batch and return
    energy production forecasts.

    This function is the entry point for the continuous-ingest scenario:
    whenever a new CSV arrives (or a DataFrame chunk is produced by a
    streaming connector), call this function with the loaded model to
    obtain predictions without retraining.

    Parameters
    ----------
    model : fitted model from `train_model` or `load_model`
    source : new data file path or DataFrame
    feature_cols : optional feature list override (must match training features)

    Returns
    -------
    pd.Series
        Predicted Energy_Generation_kW values, index-aligned to the input.
    """

    # Step 1: Preprocess new data with the same pipeline used during training
    df_new = load_and_preprocess_data(source)

    # Step 2: Select features (inference mode -- no target expected)
    X_new, _, _, _ = prepare_features(
        df_new,
        feature_cols=feature_cols,
        is_inference=True,
    )

    # Step 3: Predict
    logger.info("Running inference on %d samples...", len(X_new))
    predictions = model.predict(X_new)
    pred_series = pd.Series(predictions, index=X_new.index, name="Predicted_Energy_kW")

    # Clip predictions to physical constraints (0 <= energy <= 350 kW)
    # Mitiga a limitacao do XGBoost em extrapolar para tempestades fora do historico
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
    """
    Generate a realistic synthetic WEC dataset with the same schema as the
    real sensor CSV.  Used exclusively for the __main__ demo block.

    The target (Energy_Generation_kW) is constructed from a physics-informed
    formula so that the model has a learnable signal.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq="1h")
    wave_hs    = rng.uniform(0.5, 5.0, n_rows)       # significant wave height [m]
    wave_tp    = rng.uniform(4.0, 16.0, n_rows)      # peak period [s]
    wave_dir   = rng.uniform(180, 360, n_rows)        # wave direction [deg]
    wind_speed = rng.uniform(2.0, 20.0, n_rows)       # wind speed [m/s]
    current_sp = rng.uniform(0.0, 1.5, n_rows)        # current speed [m/s]
    air_temp   = rng.uniform(10.0, 25.0, n_rows)      # air temperature [C]
    atm_press  = rng.uniform(1000, 1025, n_rows)       # pressure [hPa]
    noise      = rng.normal(0, 10, n_rows)             # sensor noise

    # Physics-informed target: output proportional to Hs^2 * Tp, modulated by
    # a simple efficiency curve.  Real WECs follow a similar power matrix.
    wave_power_flux = wave_hs ** 2 * wave_tp
    energy = (
        0.8 * wave_power_flux
        + 1.5 * wind_speed
        + 5.0 * current_sp
        + noise
    ).clip(min=0)

    df = pd.DataFrame({
        TIMESTAMP_COL:         timestamps,
        "Buoy_ID":             "Boia_1",
        "Buoy_Latitude":        41.14,
        "Buoy_Longitude":      -8.7,
        "SST":                  rng.uniform(13, 20, n_rows),
        "Salinity":             rng.uniform(34, 36, n_rows),
        "Conductivity":         rng.uniform(50, 56, n_rows),
        "Dissolved_Oxygen":     rng.uniform(7, 10, n_rows),
        "Turbidity":            rng.uniform(0.5, 5, n_rows),
        "Chlorophyll_a":        rng.uniform(0.5, 3, n_rows),
        "Wave_Hs":              wave_hs,
        "Wave_Tp":              wave_tp,
        "Wave_Dir":             wave_dir,
        "Current_Speed":        current_sp,
        "Current_Dir":          rng.uniform(0, 360, n_rows),
        "Air_Temperature":      air_temp,
        "Atmospheric_Pressure": atm_press,
        "Relative_Humidity":    rng.uniform(60, 95, n_rows),
        "Solar_Radiation":      rng.uniform(0, 800, n_rows),
        "Wind_Speed":           wind_speed,
        "Wind_Direction":       rng.uniform(0, 360, n_rows),
        "Rainfall":             rng.uniform(0, 5, n_rows),
        "Battery_Voltage":      rng.uniform(11, 14, n_rows),
        TARGET_COL:             energy,
        "Epoch_Marker":         range(1, n_rows + 1),
    })

    # Randomly insert ~2% missing values to validate imputation logic
    for col in ["Wave_Hs", "Wave_Tp", "Wind_Speed"]:
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

    # ------------------------------------------------------------------
    # CONFIG: point to a real CSV or leave as None to use synthetic data
    # ------------------------------------------------------------------
    DATA_CSV_PATH = "datasets/wec_c5_mock_data_epochs.csv"   # set to None for synthetic demo
    MODEL_SAVE_PATH = "wec_phase1_model.joblib"

    # ------------------------------------------------------------------
    # A) TRAINING PHASE
    #    Represents the initial offline training on historical data.
    # ------------------------------------------------------------------
    logger.info("[PHASE A] Training on historical data")

    if DATA_CSV_PATH and os.path.exists(DATA_CSV_PATH):
        logger.info("Using real data file: %s", DATA_CSV_PATH)
        df_historical = load_and_preprocess_data(DATA_CSV_PATH)
    else:
        logger.info("CSV not found -- generating synthetic dataset for demo")
        raw_df = _generate_synthetic_data(n_rows=2000)
        df_historical = load_and_preprocess_data(raw_df)

    # Prepare features with temporal train/test split
    X_train, y_train, X_test, y_test = prepare_features(df_historical)

    # Train XGBoost model
    model = train_model(X_train, y_train, X_test, y_test, model_type="xgboost")

    # Persist model to disk
    save_model(model, path=MODEL_SAVE_PATH)

    # ------------------------------------------------------------------
    # B) INFERENCE PHASE (simulating arrival of a new data batch)
    #    The model is reloaded from disk ·-- simulating a cold-start
    #    scenario where training and inference run as separate processes.
    # ------------------------------------------------------------------
    logger.info("[PHASE B] Inference on a new incoming data batch")

    # Reload model (simulates a separate inference process / container)
    loaded_model = load_model(path=MODEL_SAVE_PATH)

    # Simulate a new batch: take the last 50 rows WITHOUT the target column
    # In production this batch arrives from a sensor API or a message queue.
    new_batch_raw = _generate_synthetic_data(n_rows=50, seed=99)
    new_batch_no_target = new_batch_raw.drop(columns=[TARGET_COL])

    predictions = predict_new_data(loaded_model, source=new_batch_no_target)

    # Display sample predictions
    sample = pd.DataFrame({
        TIMESTAMP_COL:             new_batch_raw[TIMESTAMP_COL].values[:10],
        "Wave_Hs":                  new_batch_raw["Wave_Hs"].values[:10],
        "Wave_Tp":                  new_batch_raw["Wave_Tp"].values[:10],
        "Predicted_Energy_kW":      predictions.values[:10],
    })

    logger.info("Sample predictions (first 10 rows):\n%s", sample.to_string(index=False))

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 60)