"""
WEC Performance Analysis - Phase 2
=====================================
Data Envelopment Analysis (DEA) for relative benchmarking of three
Wave Energy Converters (Boia_1, Boia_2, Boia_3) and early detection
of asset degradation.

DEA Model Specification
-----------------------
Type    : BCC (Banker-Charnes-Cooper) -- Variable Returns to Scale (VRS)
Orient  : Output-oriented
Input   : Wave_Power_Flux  (Hs^2 * Tp)   -- available wave energy density proxy
Output  : Energy_Generation_kW           -- actual electrical output

Rationale for single-input / single-output:
  Wave_Power_Flux encapsulates the dominant physical drivers identified in
  Phase 1 (Hs and Tp), collapsing multicollinearity into one physically
  meaningful composite.  Using it as the sole input makes the DEA result
  interpretable: "given the wave energy available, how efficiently is each
  buoy converting it to electricity?"

Solver : scipy.optimize.linprog (HiGHS backend, no external DEA library
         required).  The BCC LP formulation is implemented from scratch for
         full mathematical transparency.

Pipeline
--------
1. load_and_prepare  -- ingest CSV, recalculate Wave_Power_Flux, handle NaNs
2. build_timestamp_batches  -- group rows by timestamp, keep complete triplets
3. calculate_dea_efficiency  -- solve BCC LP for one batch of 3 DMUs
4. run_dea_timeseries  -- iterate batches, collect efficiency time series
5. aggregate_and_plot  -- weekly rolling average + save PNG
"""

import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
import seaborn as sns
matplotlib.use("Agg")  # headless backend: no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMESTAMP_COL    = "PCTimeStamp"
BUOY_ID_COL      = "Buoy_ID"
TARGET_COL       = "Energy_Generation_kW"
INPUT_COL        = "Wave_Power_Flux"
EXPECTED_BUOYS   = ["Boia_1", "Boia_2", "Boia_3"]

DATA_CSV_PATH    = "datasets/wec_c5_mock_data_epochs.csv"
PLOT_OUTPUT_PATH = "wec_phase2_dea.png"

# Degradation injection parameters for Boia_3 (used in synthetic data only)
DEGRADATION_START = "2025-05-01"
DEGRADATION_FACTOR = 0.45   # Boia_3 retains only 45% of its normal output


# ===========================================================================
# Section 1 -- Data Loading and Preparation
# ===========================================================================

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Read the sensor CSV, recalculate Wave_Power_Flux, and sanitise NaNs.

    Wave_Power_Flux is NOT stored in the raw CSV.  It is recomputed here
    using the same formula used in Phase 1:
        Wave_Power_Flux = Wave_Hs^2 * Wave_Tp

    NaN imputation uses per-column median (robust to outliers) rather than
    mean, to avoid inflating the LP inputs when wave measurements are missing.
    Using mean on skewed sea-state distributions can bias the DEA reference
    set and generate spuriously efficient frontier points.

    Parameters
    ----------
    csv_path : str
        Path to the raw sensor CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with Wave_Power_Flux column added.
    """
    logger.info("Loading data from: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=[TIMESTAMP_COL])
    logger.info("Raw shape: %s", df.shape)
    logger.info("Buoys present: %s", df[BUOY_ID_COL].unique().tolist())

    # Recalculate the composite input feature
    df[INPUT_COL] = df["Wave_Hs"] ** 2 * df["Wave_Tp"]

    # Identify numeric columns that require imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Per-column median imputation: safe for LP solvers which crash on NaN
    missing_total = df[numeric_cols].isna().sum().sum()
    if missing_total > 0:
        logger.info("Imputing %d missing values with per-column median", missing_total)
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # Sort chronologically to preserve temporal structure for rolling windows
    df = df.sort_values([TIMESTAMP_COL, BUOY_ID_COL]).reset_index(drop=True)

    logger.info("Prepared shape: %s", df.shape)
    return df


# ===========================================================================
# Section 2 -- Timestamp Batch Construction
# ===========================================================================

def build_timestamp_batches(df: pd.DataFrame) -> list[dict]:
    """
    Group sensor rows by PCTimeStamp and return only complete batches.
    
    MITIGATION: Phase Shift / Time Lag
    Applies a 3-hour rolling mean to Inputs and Outputs before batching.
    This ensures that wave travel time between buoys (up to 6km apart)
    does not create spurious inefficiencies in the strict hourly DEA.
    """
    # Aplicar a mitigacao do Phase Shift (Media Movel de 3 horas por boia)
    numeric_cols = [INPUT_COL, TARGET_COL]
    df_smoothed = df.copy()
    
    # Ordenar rigorosamente para o rolling funcionar
    df_smoothed = df_smoothed.sort_values([BUOY_ID_COL, TIMESTAMP_COL])
    
    for buoy in EXPECTED_BUOYS:
        mask = df_smoothed[BUOY_ID_COL] == buoy
        df_smoothed.loc[mask, numeric_cols] = (
            df_smoothed.loc[mask, numeric_cols]
            .rolling(window=3, min_periods=1)
            .mean()
        )

    # Voltar a ordenar por tempo para criar os batches
    df_smoothed = df_smoothed.sort_values(TIMESTAMP_COL)

    batches = []
    discarded = 0
    
    # IMPORTANTE: Mudar o iterador de 'df' para 'df_smoothed'
    for ts, group in df_smoothed.groupby(TIMESTAMP_COL):
        # Require all three buoys at every timestamp
        present_buoys = group[BUOY_ID_COL].unique().tolist()
        if not all(b in present_buoys for b in EXPECTED_BUOYS):
            discarded += 1
            continue

        # Extract values in a consistent order
        rows = [group[group[BUOY_ID_COL] == b].iloc[0] for b in EXPECTED_BUOYS]
        x_vals = np.array([r[INPUT_COL] for r in rows], dtype=float)
        y_vals = np.array([r[TARGET_COL] for r in rows], dtype=float)

        # Guard against non-positive values (LP infeasibility)
        if np.any(x_vals <= 0) or np.any(y_vals <= 0):
            discarded += 1
            continue

        batches.append({
            "timestamp": ts,
            "buoys": EXPECTED_BUOYS[:],
            "x": x_vals,
            "y": y_vals,
        })

    logger.info(
        "Built %d valid batches, discarded %d incomplete/degenerate timestamps",
        len(batches), discarded,
    )
    return batches


# ===========================================================================
# Section 3 -- BCC Output-Oriented DEA (LP formulation)
# ===========================================================================

def calculate_dea_efficiency(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve the BCC (VRS) output-oriented DEA LP for each DMU in a single batch.

    Mathematical Formulation
    ------------------------
    For each evaluated DMU k (k = 1 ... n), we solve:

        Maximise   phi_k
        subject to:
            sum_j ( lambda_j * x_ij ) <=  x_ik        for each input  i
            sum_j ( lambda_j * y_rj ) >=  phi_k * y_rk for each output r
            sum_j   lambda_j          = 1              (VRS / convexity)
            lambda_j >= 0,  phi_k unconstrained (>= 1 in practice)

    Where:
        n      = number of DMUs (3 buoys)
        lambda = peer weights (reference set intensities)
        phi    = output expansion factor (>= 1)

    The output efficiency score is:  efficiency_k = 1 / phi_k*
    A score of 1.0 means the DMU lies on the efficient frontier.
    A score of 0.7 means the DMU should produce 1/0.7 = 1.43x more output
    given the same input to be considered efficient.

    scipy.optimize.linprog minimises, so we minimise -phi.

    Decision variable vector z = [phi, lambda_1, ..., lambda_n]
    Dimension: 1 + n

    Inequality constraints (A_ub @ z <= b_ub):
        - Input  (<=):  [0, x_1, ..., x_n] @ z  <= x_k   -- one row per input
        - Output (<=):  [y_k, -y_1, ..., -y_n] @ z <= 0  -- one row per output
          (rewritten from sum_j lambda_j y_rj >= phi y_rk)

    Equality constraints (A_eq @ z = b_eq):
        - VRS:  [0, 1, ..., 1] @ z = 1

    Bounds:
        phi    : (0, None)  -- can exceed 1
        lambda : (0, None)  -- non-negative weights

    Parameters
    ----------
    x : np.ndarray shape (n,)   -- inputs  for n DMUs
    y : np.ndarray shape (n,)   -- outputs for n DMUs

    Returns
    -------
    np.ndarray shape (n,)
        Efficiency scores in [0, 1].  NaN if the LP fails for a DMU.
    """
    n = len(x)
    scores = np.full(n, np.nan)

    # Objective: minimise -phi  (phi is variable index 0)
    c = np.zeros(1 + n)
    c[0] = -1.0

    # VRS equality constraint: sum of lambdas = 1
    A_eq = np.zeros((1, 1 + n))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])

    # Bounds: phi unrestricted below (but effectively >= 1), lambdas >= 0
    # Setting phi lower bound to 0 avoids solver issues; result phi* >= 1
    bounds = [(0, None)] + [(0, None)] * n

    # Mitigacao de Outliers Falsos para o DEA: Limite Mecanico Absoluto
    # Se uma boia registar um pico absurdo, limitamos a 350kW para nao
    # distorcer a Fronteira de Possibilidades de Producao para as outras.
    y_clipped = np.clip(y, a_min=0.1, a_max=350.0)

    for k in range(n):
        # Input constraint row: [0, x_1, ..., x_n] <= x_k
        row_input = np.zeros(1 + n)
        row_input[1:] = x
        # Output constraint row (using clipped Y): phi * y_k - sum_j lambda_j * y_j <= 0
        row_output = np.zeros(1 + n)
        row_output[0]  =  y_clipped[k]   # coefficient for phi
        row_output[1:] = -y_clipped      # coefficients for lambdas

        A_ub = np.vstack([row_input, row_output])
        b_ub = np.array([x[k], 0.0])
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",         # HiGHS is the fastest and most robust LP solver in scipy
            options={"disp": False},
        )

        if result.status == 0:
            phi_star = result.x[0]
            # Clamp phi to >= 1 (numerical tolerance can produce 0.9999...)
            phi_star = max(phi_star, 1.0)
            scores[k] = 1.0 / phi_star
        else:
            # LP failed: log and leave as NaN so downstream aggregation skips it
            logger.debug(
                "LP failed for DMU %d at this batch (solver status %d: %s)",
                k, result.status, result.message,
            )

    return scores


# ===========================================================================
# Section 4 -- DEA Time-Series Runner
# ===========================================================================

def run_dea_timeseries(batches: list[dict]) -> pd.DataFrame:
    """
    Apply calculate_dea_efficiency to every valid timestamp batch and
    assemble a tidy long-format DataFrame of efficiency scores over time.

    Parameters
    ----------
    batches : list of dict from build_timestamp_batches

    Returns
    -------
    pd.DataFrame with columns [PCTimeStamp, Buoy_ID, DEA_Efficiency]
        NaN rows are retained so that the weekly rolling window correctly
        accounts for missing data (pd.Series.rolling skips NaN by default).
    """
    records = []

    for batch in batches:
        ts    = batch["timestamp"]
        buoys = batch["buoys"]
        x     = batch["x"]
        y     = batch["y"]

        scores = calculate_dea_efficiency(x, y)

        for buoy, score in zip(buoys, scores):
            records.append({
                TIMESTAMP_COL: ts,
                BUOY_ID_COL:   buoy,
                "DEA_Efficiency": score,
            })

    df_results = pd.DataFrame(records)
    logger.info(
        "DEA time series computed: %d rows, %d timestamps",
        len(df_results),
        df_results[TIMESTAMP_COL].nunique(),
    )
    return df_results

# ===========================================================================
# Section 5 -- Aggregation and Visualisation
# ===========================================================================

def aggregate_and_plot(df_results: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling means and generate two distinct visualisations:
    1. Time-series of DEA efficiency.
    2. Output vs Input Production Possibility Frontier (PPF) scatter plot.
    """
    # -----------------------------------------------------------------------
    # Prep for Plot 1: Time Series
    # -----------------------------------------------------------------------
    df_wide = df_results.pivot_table(
        index=TIMESTAMP_COL,
        columns=BUOY_ID_COL,
        values="DEA_Efficiency",
    )
    df_wide.columns.name = None
    rolling = df_wide.rolling("7D", min_periods=1).mean()
    rolling.columns = [f"{c}_roll7D" for c in rolling.columns]

    _print_degradation_summary(df_wide)

    # -----------------------------------------------------------------------
    # PLOT 1: Time Series of DEA Efficiency
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors = {"Boia_1": "#2196F3", "Boia_2": "#4CAF50", "Boia_3": "#F44336"}
    
    for buoy in EXPECTED_BUOYS:
        roll_col = f"{buoy}_roll7D"
        if roll_col in rolling.columns:
            ax1.plot(
                rolling.index,
                rolling[roll_col],
                linewidth=2.5,
                color=colors[buoy],
                label=f"{buoy} (7-Day Avg)",
            )
            
    ax1.axvline(pd.Timestamp(DEGRADATION_START), color="#FF5722", lw=2,
                linestyle="--", label="Início da Degradação (01 Maio)")
    ax1.axhspan(0.0, 0.65, alpha=0.1, color="#F44336")
    
    ax1.set_ylabel("Score de Eficiência Relativa (DEA)", fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Evolução da Eficiência DEA (Detenção de Falha)", fontweight='bold')
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    
    plt.tight_layout()
    plt.savefig("wec_phase2_dea_timeseries.png", dpi=150)
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # PLOT 2: Production Possibility Frontier (Output vs Input)
    # -----------------------------------------------------------------------
    # Juntar os scores DEA com os valores brutos para poder plotar Input vs Output
    df_frontier = pd.merge(
        df_raw[[TIMESTAMP_COL, BUOY_ID_COL, INPUT_COL, TARGET_COL, "Epoch_Marker"]],
        df_results,
        on=[TIMESTAMP_COL, BUOY_ID_COL],
        how="inner"
    )
    
    # Focar na Época 3 (onde ocorre a degradação para se notar o efeito do DEA)
    df_epoch3 = df_frontier[df_frontier["Epoch_Marker"] == 3]

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Plotar todos os pontos da Época 3
    sns.scatterplot(
        data=df_epoch3, 
        x=INPUT_COL, 
        y=TARGET_COL, 
        hue=BUOY_ID_COL, 
        palette=colors,
        alpha=0.6,
        s=40,
        ax=ax2
    )

    # Identificar e destacar a Fronteira Empírica (pontos onde DEA_Efficiency == 1)
    # Uma pequena tolerância (0.99) devido a arredondamentos matemáticos do solver
    frontier_points = df_epoch3[df_epoch3["DEA_Efficiency"] >= 0.99]
    
    # Para desenhar a linha, ordenamos pelos valores de Input
    frontier_points = frontier_points.sort_values(by=INPUT_COL)
    
    ax2.plot(
        frontier_points[INPUT_COL], 
        frontier_points[TARGET_COL], 
        color='black', 
        linestyle='-', 
        linewidth=1.5,
        label="Fronteira de Eficiência (DEA = 1.0)",
        zorder=10
    )

    ax2.set_title("Data Envelopment Analysis: Fronteira de Produção (Época 3)", fontweight='bold')
    ax2.set_xlabel("INPUT: Fluxo de Potência da Onda (Wave_Power_Flux)")
    ax2.set_ylabel("OUTPUT: Geração de Energia Real (kW)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("wec_phase2_dea_frontier.png", dpi=150)
    plt.close(fig2)

    return rolling


def _print_degradation_summary(df_wide: pd.DataFrame) -> None:
    """
    Print a terminal summary comparing pre- and post-degradation mean
    efficiency for each buoy, focussing on the Epoch 3 boundary.
    """
    cutoff = pd.Timestamp(DEGRADATION_START)
    pre  = df_wide[df_wide.index < cutoff]
    post = df_wide[df_wide.index >= cutoff]

    logger.info("")
    logger.info("=" * 62)
    logger.info("DEA DEGRADATION REPORT -- Epoch 3 boundary: %s", DEGRADATION_START)
    logger.info("=" * 62)
    logger.info("%-10s | %18s | %18s | %12s", "Buoy", "Pre-May Mean", "Post-May Mean", "Delta")
    logger.info("-" * 62)
    for buoy in EXPECTED_BUOYS:
        if buoy not in df_wide.columns:
            continue
        pre_mean  = pre[buoy].mean()
        post_mean = post[buoy].mean()
        delta     = post_mean - pre_mean
        flag      = "  <<< DEGRADATION DETECTED" if delta < -0.15 else ""
        logger.info(
            "%-10s | %18.4f | %18.4f | %+12.4f%s",
            buoy, pre_mean, post_mean, delta, flag,
        )
    logger.info("=" * 62)
    logger.info("")


# ===========================================================================
# Synthetic Data Generator (fallback when CSV is absent)
# ===========================================================================

def _generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic sensor dataset for three WEC buoys spanning three
    epochs (March 2025 to June 2025), with a realistic degradation pattern
    injected for Boia_3 starting on 01 May 2025.

    Epoch structure:
        Epoch 1  --  2025-03-01 to 2025-03-31  (all buoys healthy)
        Epoch 2  --  2025-04-01 to 2025-04-30  (all buoys healthy)
        Epoch 3  --  2025-05-01 to 2025-06-15  (Boia_3 degraded)

    Boia_3 in Epoch 3 produces only DEGRADATION_FACTOR (45%) of what a
    healthy converter would produce under the same wave conditions.  This
    simulates a mechanical fault (e.g., seal failure, PTO dysfunction)
    causing a sudden and persistent drop in conversion efficiency.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2025-03-01", "2025-06-15", freq="1h")
    n = len(timestamps)

    rows = []
    for buoy_id in EXPECTED_BUOYS:
        # Shared sea-state variables (same conditions for all buoys per timestamp)
        wave_hs   = rng.uniform(0.5, 5.0,  n)
        wave_tp   = rng.uniform(4.0, 16.0, n)
        wave_dir  = rng.uniform(180, 360,  n)
        wind_spd  = rng.uniform(2.0, 20.0, n)
        curr_spd  = rng.uniform(0.0, 1.5,  n)
        wind_dir  = rng.uniform(0,   360,  n)
        air_temp  = rng.uniform(10,  25,   n)
        atm_press = rng.uniform(1000, 1025, n)

        # Healthy output model: physics-informed (same as Phase 1 synthetic)
        wave_power_flux = wave_hs ** 2 * wave_tp
        noise  = rng.normal(0, 8, n)
        energy = (0.85 * wave_power_flux + 1.5 * wind_spd + 5.0 * curr_spd + noise).clip(min=1.0)

        # Inject degradation for Boia_3 in Epoch 3
        if buoy_id == "Boia_3":
            deg_mask = timestamps >= DEGRADATION_START
            energy[deg_mask] *= DEGRADATION_FACTOR
            # Add extra noise to degrade signal quality, simulating sensor drift
            energy[deg_mask] += rng.normal(0, 5, deg_mask.sum())
            energy = energy.clip(min=0.5)

        # Epoch label (1, 2, 3) based on month
        epoch_markers = np.where(
            timestamps < "2025-04-01", 1,
            np.where(timestamps < "2025-05-01", 2, 3)
        )

        df_buoy = pd.DataFrame({
            TIMESTAMP_COL:         timestamps,
            BUOY_ID_COL:           buoy_id,
            "Buoy_Latitude":        41.14,
            "Buoy_Longitude":      -8.7,
            "SST":                  rng.uniform(13, 20, n),
            "Salinity":             rng.uniform(34, 36, n),
            "Conductivity":         rng.uniform(50, 56, n),
            "Dissolved_Oxygen":     rng.uniform(7, 10, n),
            "Turbidity":            rng.uniform(0.5, 5, n),
            "Chlorophyll_a":        rng.uniform(0.5, 3, n),
            "Wave_Hs":              wave_hs,
            "Wave_Tp":              wave_tp,
            "Wave_Dir":             wave_dir,
            "Current_Speed":        curr_spd,
            "Current_Dir":          rng.uniform(0, 360, n),
            "Air_Temperature":      air_temp,
            "Atmospheric_Pressure": atm_press,
            "Relative_Humidity":    rng.uniform(60, 95, n),
            "Solar_Radiation":      rng.uniform(0, 800, n),
            "Wind_Speed":           wind_spd,
            "Wind_Direction":       wind_dir,
            "Rainfall":             rng.uniform(0, 5, n),
            "Battery_Voltage":      rng.uniform(11, 14, n),
            TARGET_COL:             energy,
            "Epoch_Marker":         epoch_markers,
        })

        # Randomly insert ~2% missing values to validate imputation
        for col in ["Wave_Hs", "Wave_Tp", "Wind_Speed", TARGET_COL]:
            mask = rng.random(n) < 0.02
            df_buoy.loc[mask, col] = np.nan

        rows.append(df_buoy)

    df_all = pd.concat(rows, ignore_index=True)
    logger.info(
        "Synthetic dataset generated: %d rows, %d timestamps, %d buoys",
        len(df_all),
        df_all[TIMESTAMP_COL].nunique(),
        df_all[BUOY_ID_COL].nunique(),
    )
    return df_all


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":

    logger.info("=" * 62)
    logger.info("WEC Phase 2 -- DEA Benchmarking and Degradation Detection")
    logger.info("=" * 62)

    # ------------------------------------------------------------------
    # Step 0: Ingest data
    # Load from real CSV if it exists; otherwise fall back to synthetic
    # data that has a known degradation pattern for validation purposes.
    # ------------------------------------------------------------------
    if os.path.exists(DATA_CSV_PATH):
        logger.info("CSV found -- loading real data from: %s", DATA_CSV_PATH)
        df_raw = pd.read_csv(DATA_CSV_PATH)
        # The real CSV does not have Wave_Power_Flux; load_and_prepare adds it
        df_prepared = load_and_prepare(DATA_CSV_PATH)
    else:
        logger.warning(
            "CSV not found at '%s'. Generating synthetic dataset with "
            "injected Boia_3 degradation for demonstration.",
            DATA_CSV_PATH,
        )
        df_raw = _generate_synthetic_data(seed=42)
        # Save synthetic CSV so subsequent runs use it directly
        os.makedirs("datasets", exist_ok=True)
        df_raw.to_csv(DATA_CSV_PATH, index=False)
        logger.info("Synthetic CSV saved to: %s", DATA_CSV_PATH)
        df_prepared = load_and_prepare(DATA_CSV_PATH)

    # ------------------------------------------------------------------
    # Step 1: Build timestamp batches
    # Only keep timestamps where all three buoys have valid measurements.
    # ------------------------------------------------------------------
    batches = build_timestamp_batches(df_prepared)

    if len(batches) == 0:
        logger.error(
            "No valid batches found. Verify that Boia_1, Boia_2, and Boia_3 "
            "share at least one common timestamp with positive Wave_Power_Flux "
            "and Energy_Generation_kW values."
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Step 2: Run DEA across the entire time series
    # ------------------------------------------------------------------
    df_results = run_dea_timeseries(batches)

    # Report raw per-epoch average scores per buoy
    df_prepared_ts = df_prepared[[TIMESTAMP_COL, "Epoch_Marker", BUOY_ID_COL]].drop_duplicates()
    df_merged = df_results.merge(df_prepared_ts, on=[TIMESTAMP_COL, BUOY_ID_COL], how="left")

    logger.info("")
    logger.info("Mean DEA Efficiency by Buoy and Epoch:")
    epoch_summary = (
        df_merged.groupby(["Epoch_Marker", BUOY_ID_COL])["DEA_Efficiency"]
        .mean()
        .unstack(BUOY_ID_COL)
        .round(4)
    )
    for epoch, row in epoch_summary.iterrows():
        row_str = "  |  ".join(f"{b}: {v:.4f}" for b, v in row.items())
        logger.info("  Epoch %s  -->  %s", epoch, row_str)

    # ------------------------------------------------------------------
    # Step 3: Aggregate and plot
    # ------------------------------------------------------------------
    rolling_df = aggregate_and_plot(df_results, df_raw=df_prepared)

    # Final confirmation in terminal
    logger.info("")
    logger.info("Phase 2 pipeline complete.")
    logger.info("Plot saved to: %s", Path(PLOT_OUTPUT_PATH).resolve())
    logger.info("")
    logger.info(
        "Interpretation: if Boia_3 shows a sustained DEA efficiency drop "
        "after %s, a physical inspection of the PTO system and "
        "mooring is recommended.",
        DEGRADATION_START,
    )