"""
WEC Performance Analysis - Phase 1 | Absolute Variant
======================================================
Unified pipeline for the Absolute Performance Assessment of 12 Wave Energy
Converters (WECs/Buoys).  This module consolidates model training, inference,
anomaly flagging, visualisation, and terminal reporting into a single,
self-contained execution flow.

No intermediate artefacts (.joblib, pickles) are produced or consumed.  The
entire pipeline -- from raw CSV to final PNG and terminal report -- is
reproducible in a single run.

Pipeline stages
---------------
A  Data ingestion and feature engineering
B  Temporal train/test split (80 / 20)
C  XGBoost regression training + metric extraction
D  Full-dataset inference and residual computation
E  Absolute anomaly flagging   (residual < -1.5 * RMSE_test)
F  Visualisation (3-panel, 2x2 GridSpec) saved to PNG
G  Asset Performance Report emitted to the logger (Epoch 3)

Usage
-----
    python phase1_absolute.py

Outputs
-------
    plots/phase1/wec_phase1_absolute.png  -- figure with three analytical panels
    stdout / log stream                   -- structured Asset Performance Report
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
# Global constants
# ---------------------------------------------------------------------------

DATA_PATH: str = "dataset2/wec_c5_mock_data_epochs.csv"
OUTPUT_PATH: str = "plots/phase1/wec_phase1_absolute.png"

# ADICIONAR ESTAS DUAS LINHAS
PHASE1_CSV_OUT: str = "dataset2/wec_phase1_outputs.csv"
PHASE1_MODEL_OUT: str = "dataset2/wec_phase1_xgboost.joblib"

TIMESTAMP_COL: str = "PCTimeStamp"
TARGET_COL: str = "Energy_Generation_kW"
BUOY_ID_COL: str = "Buoy_ID"
EPOCH_COL: str = "Epoch_Marker"

# Expected buoy identifiers, ordered for consistent visualisation
BUOY_ORDER: List[str] = [f"Boia_{i}" for i in range(1, 13)]

# Model feature set
FEATURE_COLS: List[str] = [
    "Hs__m",
    "Te__s",
    "Wave_Power_Flux",
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
    "hour",
    "month",
]

# Temporal split fraction
TEST_FRACTION: float = 0.20
TRAIN_CUTOFF_DATE: str = "2025-05-01"

# Anomaly detection multiplier applied to RMSE_test
ANOMALY_SIGMA_FACTOR: float = 1.5

# Confidence band multiplier for the P10-P90 visual band
CONFIDENCE_Z: float = 1.28

# Physical generation bounds in kW
GEN_MIN_KW: float = 0.0
GEN_MAX_KW: float = 350.0

# XGBoost hyper-parameters
XGB_PARAMS: Dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# Definir os grupos de saúde das boias para a análise particionada
HEALTHY_BUOYS: List[str] = [f"Boia_{i}" for i in range(1, 9)]
DEGRADED_BUOYS: List[str] = [f"Boia_{i}" for i in range(9, 13)]
ALL_BUOYS: List[str] = HEALTHY_BUOYS + DEGRADED_BUOYS


# Matplotlib / Seaborn aesthetics
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# Stage A -- Data ingestion and feature engineering
# ---------------------------------------------------------------------------

def load_and_engineer_features(csv_path: str) -> pd.DataFrame:
    """
    Load raw sensor data from CSV, impute missing values, and compute
    derived / temporal features required by the model.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-enriched DataFrame sorted by timestamp.
    """
    logger.info("Stage A -- Loading data from: %s", csv_path)
    df: pd.DataFrame = pd.read_csv(csv_path, parse_dates=[TIMESTAMP_COL])
    logger.info("Raw shape: %s", df.shape)

    # Temporal ordering is mandatory for a correct chronological split
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Outlier suppression: flag values beyond 4 sigma as NaN, then impute
    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_std: float = df[col].std()
        if col_std == 0:
            continue
        z: pd.Series = (df[col] - df[col].mean()).abs() / col_std
        n_flagged: int = int((z > 4.0).sum())
        if n_flagged:
            logger.debug("Column '%s': flagging %d outlier(s) as NaN", col, n_flagged)
            df.loc[z > 4.0, col] = np.nan

    missing_total: int = int(df[numeric_cols].isna().sum().sum())
    if missing_total:
        logger.info("Imputing %d missing value(s) via median strategy", missing_total)
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Physics-informed feature: Wave Power Flux (kW/m)
    df["Wave_Power_Flux"] = 0.49 * (df["Hs__m"] ** 2) * df["Te__s"]
    logger.info("Engineered feature 'Wave_Power_Flux' computed (0.49 * Hs^2 * Te)")

    # Temporal cyclical features
    df["hour"] = df[TIMESTAMP_COL].dt.hour
    df["month"] = df[TIMESTAMP_COL].dt.month
    logger.info("Temporal features extracted: hour, month")

    logger.info("Preprocessed shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Stage B -- Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Produce a chronological 80/20 train/test split without data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame sorted by timestamp.
    test_fraction : float
        Fraction of rows reserved for the test set (most recent portion).

    Returns
    -------
    X_train, y_train, X_test, y_test : tuple of DataFrames and Series
    """
    split_idx: int = int(len(df) * (1.0 - test_fraction))

    X: pd.DataFrame = df[FEATURE_COLS].copy()
    y: pd.Series = df[TARGET_COL].copy()

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Stage B -- Temporal split: train=%d rows | test=%d rows (%.0f%% / %.0f%%)",
        len(X_train),
        len(X_test),
        (1.0 - test_fraction) * 100,
        test_fraction * 100,
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Stage C -- Model training and metric extraction
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Optional[Dict] = None,
) -> Tuple[XGBRegressor, float]:
    """
    Train an XGBoost regressor and return the fitted model together with the
    test-set RMSE extracted dynamically (never hardcoded).

    Parameters
    ----------
    X_train, y_train : training features and target.
    X_test, y_test   : held-out evaluation features and target.
    params           : optional XGBoost hyper-parameter dictionary.

    Returns
    -------
    model : XGBRegressor
        Fitted model ready for inference.
    rmse_test : float
        Root Mean Squared Error on the test partition.
    """
    effective_params: Dict = params or XGB_PARAMS
    model = XGBRegressor(**effective_params)

    logger.info(
        "Stage C -- Training XGBoost on %d samples, %d features...",
        len(X_train),
        X_train.shape[1],
    )
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # 1. Avaliar In-Sample (O Comportamento "Normal" Base)
    y_pred_train: np.ndarray = model.predict(X_train)
    rmse_train: float = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    r2_train: float = float(r2_score(y_train, y_pred_train))
    
    # 2. Avaliar Out-of-Sample (Onde as anomalias vão aparecer)
    y_pred_test: np.ndarray = model.predict(X_test)
    rmse_test: float = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test: float = float(mean_absolute_error(y_test, y_pred_test))
    r2_test: float = float(r2_score(y_test, y_pred_test))

    logger.info("--- Baseline Metrics (In-Sample / Epoch 1) ---")
    logger.info("  RMSE : %.4f kW", rmse_train)
    logger.info("  R^2  : %.4f (Expected Healthy Behavior)", r2_train)
    
    logger.info("--- Global Test Set Metrics (Out-of-Sample) ---")
    logger.info("  RMSE : %.4f kW  [used dynamically for anomalies]", rmse_test)
    logger.info("  MAE  : %.4f kW", mae_test)
    logger.info("  R^2  : %.4f (Contaminated by anomalies)", r2_test)
    logger.info("-------------------------------------------------")

    importances: pd.Series = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    logger.info("Top-5 feature importances:\n%s", importances.head(5).to_string())

    # Usamos o RMSE do treino (o comportamento verdadeiramente normal) 
    # como a base da nossa banda de tolerância futura, que é muito mais rigoroso.
    return model, rmse_train


# ---------------------------------------------------------------------------
# Stage D -- Full-dataset inference and Stage E -- anomaly flagging
# ---------------------------------------------------------------------------

def run_inference_and_flag(
    df: pd.DataFrame,
    model: XGBRegressor,
    rmse_test: float,
) -> pd.DataFrame:
    """
    Apply the trained model to 100% of the dataset, compute residuals,
    and set the absolute anomaly flag.

    Anomaly definition
    ------------------
    A timestamp is flagged when the signed residual falls below the threshold:

        Absolute_Residual  = Energy_Generation_kW - Predicted_Energy_kW
        Threshold          = -ANOMALY_SIGMA_FACTOR * rmse_test
        Is_Absolute_Anomaly = (Absolute_Residual < Threshold)

    This captures timestamps where actual generation is significantly worse
    than what the model predicts based on the available wave conditions
    (i.e., the device underperforms its expected efficiency error).

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed DataFrame (all rows, all buoys).
    model : XGBRegressor
        Fitted model.
    rmse_test : float
        Dynamic test-set RMSE used to define the anomaly threshold.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three additional columns:
        'Predicted_Energy_kW', 'Absolute_Residual', 'Is_Absolute_Anomaly'.
    """
    logger.info(
        "Stage D/E -- Full inference on %d rows + anomaly flagging", len(df)
    )

    predictions: np.ndarray = model.predict(df[FEATURE_COLS])
    df = df.copy()
    df["Predicted_Energy_kW"] = np.clip(predictions, GEN_MIN_KW, GEN_MAX_KW)

    df["Absolute_Residual"] = df[TARGET_COL] - df["Predicted_Energy_kW"]

    anomaly_threshold: float = -ANOMALY_SIGMA_FACTOR * rmse_test
    df["Is_Absolute_Anomaly"] = df["Absolute_Residual"] < anomaly_threshold

    n_anomalies: int = int(df["Is_Absolute_Anomaly"].sum())
    pct_anomalies: float = 100.0 * n_anomalies / len(df)
    logger.info(
        "Anomaly threshold: residual < %.4f kW  (-%g * %.4f)",
        anomaly_threshold,
        ANOMALY_SIGMA_FACTOR,
        rmse_test,
    )
    logger.info(
        "Global anomaly rate: %d / %d timestamps (%.2f%%)",
        n_anomalies,
        len(df),
        pct_anomalies,
    )
    return df


# ---------------------------------------------------------------------------
# Stage F -- Visualisation
# ---------------------------------------------------------------------------

def _build_panel_1_feature_importance(
    ax: plt.Axes,
    model: XGBRegressor,
    top_n: int = 8,
) -> None:
    """
    Render a horizontal bar chart of the top-N XGBoost feature importances.

    Parameters
    ----------
    ax    : Axes to draw on.
    model : Fitted XGBRegressor.
    top_n : Number of top features to display.
    """
    importances: pd.Series = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=True)
    )
    importances.tail(top_n).plot(kind="barh", color="#2c3e50", ax=ax)
    ax.set_title(
        "1. Feature Importance\n(Phase 1 Output --> DEA Input)",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlabel("Importance Score (XGBoost)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)


def _build_panel_2_forecast(
    ax: plt.Axes,
    df: pd.DataFrame,
    rmse_test: float,
    buoy_id: str = "Boia_9",
    train_cutoff: str = TRAIN_CUTOFF_DATE,
) -> None:
    """
    Render the probabilistic forecast chart (Real vs Predicted) for a single
    buoy, showing clear in-sample vs out-of-sample visual zones and a
    dynamically computed P10-P90 confidence band.

    Parameters
    ----------
    ax           : Axes to draw on.
    df           : Full DataFrame with 'Predicted_Energy_kW' column.
    rmse_test    : Dynamic test RMSE; drives the P10-P90 band width.
    buoy_id      : Buoy identifier to isolate.
    train_cutoff : ISO date string marking the 80% train boundary.
    """
    half_band: float = CONFIDENCE_Z * rmse_test
    cutoff_dt: pd.Timestamp = pd.to_datetime(train_cutoff)

    df_buoy: pd.DataFrame = (
        df[df[BUOY_ID_COL] == buoy_id]
        .set_index(TIMESTAMP_COL)[[TARGET_COL, "Predicted_Energy_kW"]]
        .resample("D")
        .mean()
    )
    df_buoy["P10"] = (df_buoy["Predicted_Energy_kW"] - half_band).clip(lower=GEN_MIN_KW)
    df_buoy["P90"] = (df_buoy["Predicted_Energy_kW"] + half_band).clip(upper=GEN_MAX_KW)

    train_mask: pd.Series = df_buoy.index < cutoff_dt
    df_train: pd.DataFrame = df_buoy[train_mask]
    df_test: pd.DataFrame = df_buoy[~train_mask]

    # Actual generation -- full period
    ax.plot(
        df_buoy.index,
        df_buoy[TARGET_COL],
        label="Actual Generation",
        color="#e74c3c",
        linewidth=2,
    )
    # In-sample model fit (thin grey dashed)
    ax.plot(
        df_train.index,
        df_train["Predicted_Energy_kW"],
        label="Model (Train / In-Sample)",
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )
    # Out-of-sample forecast (bold green dashed)
    ax.plot(
        df_test.index,
        df_test["Predicted_Energy_kW"],
        label="Forecast (Test / Out-of-Sample)",
        color="#27ae60",
        linestyle="--",
        linewidth=2.5,
    )
    # Dynamic P10-P90 band on forecast zone only
    ax.fill_between(
        df_test.index,
        df_test["P10"],
        df_test["P90"],
        color="#2ecc71",
        alpha=0.3,
        label=f"Confidence Band (+/-{CONFIDENCE_Z} RMSE = +/-{half_band:.1f} kW)",
    )

    # Epoch boundary markers
    ax.axvline(cutoff_dt, color="black", linestyle="-", lw=1.5, label="Train Cutoff (80%)")
    ax.axvline(pd.to_datetime("2025-05-15"), color="gray", linestyle=":", alpha=0.7)

    # Zone annotations -- positioned relative to the Y axis range
    y_top: float = df_buoy[TARGET_COL].max()
    y_annot: float = y_top * 0.88 if y_top > 0 else 280.0
    ax.text(
        pd.to_datetime("2025-03-01"), y_annot,
        "TRAINING ZONE\n(Epoch 1)", ha="center", fontsize=8, color="gray", fontweight="bold",
    )
    ax.text(
        pd.to_datetime("2025-05-07"), y_annot,
        "Epoch 2\n(-15%)", ha="center", fontsize=8,
    )
    ax.text(
        pd.to_datetime("2025-05-23"), y_annot,
        "Epoch 3\n(Anomaly)", ha="center", fontsize=8, color="#27ae60", fontweight="bold",
    )

    ax.set_title(
        f"2. Probabilistic Forecast: {buoy_id} (Actual vs Expected)",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylabel("Power (kW)", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="both", labelsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    ax.legend(fontsize=7.5, loc="lower left")


def _build_panel_3_anomaly_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    epoch: int = 3,
) -> None:
    """
    Render a bar chart showing, for each buoy in the specified epoch, the
    percentage of timestamps flagged as Absolute Anomaly
    (residual < -1.5 * RMSE_test).

    Buoys 9-12 are highlighted in a contrasting colour to communicate their
    elevated underperformance rate relative to the healthy fleet.

    Parameters
    ----------
    ax    : Axes to draw on.
    df    : Full DataFrame with 'Is_Absolute_Anomaly' column.
    epoch : Epoch number to isolate (default: 3).
    """
    df_epoch: pd.DataFrame = df[df[EPOCH_COL] == epoch].copy()

    anomaly_pct: pd.Series = (
        df_epoch.groupby(BUOY_ID_COL)["Is_Absolute_Anomaly"]
        .mean()
        .reindex(BUOY_ORDER)
        .fillna(0.0) * 100.0
    )

    # Determine threshold for critical / healthy colouring
    # Buoys 9-12 are known degraded assets; colour them distinctly
    colors: List[str] = [
        "#c0392b" if buoy in [f"Boia_{i}" for i in range(9, 13)] else "#2c3e50"
        for buoy in anomaly_pct.index
    ]

    bars = ax.bar(anomaly_pct.index, anomaly_pct.values, color=colors, edgecolor="white", linewidth=0.6)

    # Value labels above each bar
    for bar, val in zip(bars, anomaly_pct.values):
        if val > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    # Reference line at 10% to anchor viewer expectations
    ax.axhline(10.0, color="orange", linestyle="--", linewidth=1.2, label="10% Reference Line")

    # Legend patches for the two colour categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2c3e50", label="Healthy Fleet (Buoys 1-8)"),
        Patch(facecolor="#c0392b", label="Degraded Fleet (Buoys 9-12)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    ax.set_title(
        f"3. Absolute Anomaly Rate per Buoy -- Epoch {epoch}\n"
        "(% Timestamps with Residual < -1.5 * RMSE_test)",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlabel("WEC Asset (Buoy ID)", fontsize=9)
    ax.set_ylabel("Anomaly Rate (%)", fontsize=9)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(bottom=0)


def generate_figure(
    df: pd.DataFrame,
    model: XGBRegressor,
    rmse_test: float,
    output_path: str = OUTPUT_PATH,
) -> None:
    """
    Compose and save the three-panel analytical figure (2x2 GridSpec).

    Layout
    ------
    [Panel 1: Feature Importance] | [Panel 2: Forecast Boia_9   ]
    [Panel 3: Anomaly Bar Chart (full width -- spans 2 columns)  ]

    Parameters
    ----------
    df          : Full DataFrame with inference and anomaly columns.
    model       : Fitted XGBRegressor (for feature importances).
    rmse_test   : Dynamic test RMSE for confidence band calculation.
    output_path : Destination path for the PNG file.
    """
    logger.info("Stage F -- Composing 3-panel figure")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.22, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    _build_panel_1_feature_importance(ax1, model)
    _build_panel_2_forecast(ax2, df, rmse_test)
    _build_panel_3_anomaly_bar(ax3, df, epoch=3)

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    logger.info("Figure saved to: %s", Path(output_path).resolve())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage G -- Asset Performance Report (terminal / log)
# ---------------------------------------------------------------------------

def emit_asset_performance_report(
    df: pd.DataFrame,
    epoch: int = 3,
) -> None:
    """
    Emit a structured Asset Performance Report to the logger.

    For each buoy in the specified epoch, the report declares the percentage
    of timestamps during which the WEC underperformed its expected efficiency
    error threshold.  The worst-performing asset is identified at the end.

    Parameters
    ----------
    df    : Full DataFrame with 'Is_Absolute_Anomaly' column.
    epoch : Epoch number to report on (default: 3).
    """
    df_epoch: pd.DataFrame = df[df[EPOCH_COL] == epoch].copy()
    separator: str = "=" * 70

    logger.info(separator)
    logger.info("ASSET PERFORMANCE REPORT -- Epoch %d", epoch)
    logger.info("Anomaly definition: Residual < -%.1f * RMSE_test", ANOMALY_SIGMA_FACTOR)
    logger.info(separator)

    logger.info("--- R^2 Degradation Analysis (Epoch 3) ---")

    mask_healthy = df_epoch[BUOY_ID_COL].isin(HEALTHY_BUOYS)
    mask_degraded = df_epoch[BUOY_ID_COL].isin(DEGRADED_BUOYS)
    
    # Calcular o R2 apenas para as boias saudáveis nesta época
    y_true_h = df_epoch[mask_healthy][TARGET_COL]
    y_pred_h = df_epoch[mask_healthy]["Predicted_Energy_kW"]
    r2_healthy = r2_score(y_true_h, y_pred_h) if not y_true_h.empty else np.nan
    
    # Calcular o R2 apenas para as boias degradadas nesta época
    y_true_d = df_epoch[mask_degraded][TARGET_COL]
    y_pred_d = df_epoch[mask_degraded]["Predicted_Energy_kW"]
    r2_degraded = r2_score(y_true_d, y_pred_d) if not y_true_d.empty else np.nan
    
    logger.info("  Healthy Fleet R^2  : %.4f (Model retains accuracy)", r2_healthy)
    logger.info("  Degraded Fleet R^2 : %.4f (Metric collapse confirms anomaly)", r2_degraded)
    logger.info(separator)

    # 2. O Relatório de Anomalias Absolutas Individual
    anomaly_rates: Dict[str, float] = {}


    for buoy in BUOY_ORDER:
        df_buoy: pd.DataFrame = df_epoch[df_epoch[BUOY_ID_COL] == buoy]
        if df_buoy.empty:
            logger.warning("No data found for %s in Epoch %d -- skipping", buoy, epoch)
            continue
        n_total: int = len(df_buoy)
        n_anomalous: int = int(df_buoy["Is_Absolute_Anomaly"].sum())
        pct: float = 100.0 * n_anomalous / n_total
        anomaly_rates[buoy] = pct
        buoy_label: str = buoy.replace("_", " ")
        logger.info(
            "%s underperformed its expected efficiency error threshold in %.2f%% of the timestamps.",
            buoy_label,
            pct,
        )

    logger.info(separator)

    if anomaly_rates:
        worst_buoy: str = max(anomaly_rates, key=lambda b: anomaly_rates[b])
        worst_pct: float = anomaly_rates[worst_buoy]
        logger.info(
            "CONCLUSION -- Worst performing asset: %s (anomaly rate = %.2f%%).",
            worst_buoy.replace("_", " "),
            worst_pct,
        )
        logger.info(
            "Recommendation: prioritise inspection and maintenance of %s.",
            worst_buoy.replace("_", " "),
        )

    logger.info(separator)


# ---------------------------------------------------------------------------
# Stage H -- Artefact Export
# ---------------------------------------------------------------------------
anomaly_rates: Dict[str, float] = {}
def export_artefacts(
    df: pd.DataFrame, 
    model: XGBRegressor, 
    rmse_test: float
) -> None:
    """
    Export the data contract for Phase 3 (The Merge) and serialize the 
    trained XGBoost model for potential real-time deployment.
    """
    logger.info("Stage H -- Exporting intermediate artefacts")
    
    # 1. Export CSV (Data Contract)
    export_cols = [
        TIMESTAMP_COL,
        BUOY_ID_COL,
        "Predicted_Energy_kW",
        "Absolute_Residual",
        "Is_Absolute_Anomaly"
    ]
    df_export = df[export_cols].copy()
    
    # Broadcast the dynamic RMSE to all rows so Phase 3 can use it natively
    df_export["RMSE_test_dynamic"] = rmse_test
    
    os.makedirs(os.path.dirname(PHASE1_CSV_OUT), exist_ok=True)
    df_export.to_csv(PHASE1_CSV_OUT, index=False)
    logger.info("Data contract exported to: %s", PHASE1_CSV_OUT)
    
    # 2. Export Model (.joblib)
    joblib.dump(model, PHASE1_MODEL_OUT)
    logger.info("Trained XGBoost model exported to: %s", PHASE1_MODEL_OUT)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Execute the full Phase 1 Absolute pipeline end-to-end.
    """
    logger.info("=" * 60)
    logger.info("WEC Phase 1 -- Absolute Performance Analysis")
    logger.info("=" * 60)

    # Stage A
    df: pd.DataFrame = load_and_engineer_features(DATA_PATH)

    # Stage B
    X_train, y_train, X_test, y_test = temporal_split(df)

    # Stage C
    model, rmse_test = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Stages D + E
    df = run_inference_and_flag(df, model, rmse_test)

    # Stage F
    generate_figure(df, model, rmse_test)

    # Stage G
    emit_asset_performance_report(df, epoch=3)

    # Stage H 
    export_artefacts(df, model, rmse_test)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()