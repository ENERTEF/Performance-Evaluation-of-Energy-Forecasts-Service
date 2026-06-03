"""
WEC Performance Analysis - Phase 2 | Stochastic Frontier Analysis (SFA)
========================================================================
Stochastic Frontier Analysis for technical efficiency scoring and
mechanical degradation detection across 12 Wave Energy Converters (WECs).

Motivation
----------
Phase 1 (XGBoost regression) identifies the absolute deviation of each
WEC from its model-predicted output.  Phase 2 (DEA) identifies
inefficiency deterministically: every deviation from the empirical
frontier is labelled as waste.  In a marine environment this is a serious
flaw because random sea-state variation, sensor noise, and measurement
error are genuinely symmetric and cannot be attributed to the asset.

SFA addresses thiOUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epoch_df = df[df["Epoch_Marker"] == REPORT_EPOCH].copy()
    if epoch_df.empty:
        logger.warning(
            "No Epoch %d data available; visualisation skipped.", REPORT_EPOCH
        )
        return

    logger.info(
        "Generating visualisation for Epoch %d: %d rows.", REPORT_EPOCH, len(epoch_df)
    )

    # Build pivot tables for both fleet groups
    healthy_states = _build_state_pivot(epoch_df, HEALTHY_FLEET)
    healthy_alarms = _build_alarm_pivot(epoch_df, HEALTHY_FLEET)
    degraded_states = _build_state_pivot(epoch_df, DEGRADED_FLEET)
    degraded_alarms = _build_alarm_pivot(epoch_df, DEGRADED_FLEET)

    # Figure layout: 2 rows (healthy / degraded), 1 column
    n_healthy = len(healthy_states.columns)
    n_degraded = len(degraded_states.columns)
    # Row height proportional to number of buoys; minimum 2 inches per panel
    row_h_healthy = max(2.0, n_healthy * 0.55)
    row_h_degraded = max(2.0, n_degraded * 0.55)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(18, row_h_healthy + row_h_degraded + 3),
        gridspec_kw={"height_ratios": [n_healthy, n_degraded]},
        facecolor="#1a1a2e",
    )

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    _draw_heatmap_panel(
        axes[0],
        healthy_states,
        healthy_alarms,
        f"Epoch {REPORT_EPOCH} - Healthy Fleet (Buoys 1-8) - Operational State Matrix",
    )
    _draw_heatmap_panel(
        axes[1],
        degraded_states,
        degraded_alarms,
        f"Epoch {REPORT_EPOCH} - Degraded Fleet (Buoys 9-12) - Operational State Matrix",
    )

    # Shared legend
    legend_elements = [
        Patch(facecolor=c, edgecolor="white", label=lbl)
        for c, lbl in zip(STATE_COLORS, STATE_LABELS)
    ]
    legend_elements.append(
        Patch(facecolor=ALARM_COLOR, alpha=0.75, edgecolor="white",
              label="Maintenance Alarm Active")
    )
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=8.5,
        framealpha=0.15,
        facecolor="#2c2c54",
        edgecolor="white",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )

    # Global title
    fig.suptitle(
        f"WEC Phase 3 Decision Matrix  |  Epoch {REPORT_EPOCH}  |  "
        "Healthy vs. Degraded Fleet Comparison",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=0.99,
    )

    # Style axes text for dark background
    for ax in axes:
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Visualisation saved to: %s", OUTPUT_PLOT)s by decomposing the composite residual into two
statistically distinct components:

    epsilon_i = v_i - u_i

where:
    v_i ~ N(0, sigma_v^2)     symmetric noise (waves, sensors)
    u_i ~ |N(0, sigma_u^2)|   one-sided technical inefficiency

Only u captures mechanical degradation.  A healthy buoy operating in
rough seas will show large v but small u; a buoy with a failed PTO will
show large u regardless of wave conditions.

Model specification: 1D Cobb-Douglas (log-linear)
--------------------------------------------------
    ln(Y_i) = beta_0 + beta_1 * ln(WPF_i) + v_i - u_i

    WPF  = Wave_Power_Flux  = 0.49 * Hs^2 * Te   [input]
    Y    = Energy_Generation_kW                   [output, capped at 350 kW]

Parameters estimated by Maximum Likelihood Estimation (MLE):
    beta_0  : intercept
    beta_1  : output elasticity with respect to wave power flux
    lambda  : sigma_u / sigma_v  (signal-to-noise ratio)
    sigma^2 : sigma_u^2 + sigma_v^2  (total error variance)

Efficiency estimator: Battese & Coelli (1988), derived from the Jondrow
et al. (1982) conditional distribution of u given epsilon:

    TE_i = E[exp(-u_i) | epsilon_i]
         = exp(-mu_star_i + sigma_star^2 / 2)
           * Phi(mu_star_i / sigma_star - sigma_star)
           / Phi(mu_star_i / sigma_star)

    where:
        mu_star_i  = -epsilon_i * sigma_u^2 / sigma^2
        sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2

Training strategy: MLE is fitted exclusively on Epoch 1 (Golden Period)
where all buoys are known to operate without mechanical faults.  This
ensures the estimated frontier represents a genuine best-practice
production surface, not a contaminated average.  Epochs 2 and 3 are
scored against this frozen frontier.

Pipeline
--------
    1. load_and_prepare           -- ingest CSV, compute WPF, apply log transform
    2. fit_sfa_epoch1             -- MLE on Epoch 1 only
    3. score_efficiency           -- Battese-Coelli estimator on full dataset
    4. compute_generation_deficit -- back-transform frontier, compute kW deficit
    5. aggregate_rolling          -- 7-day rolling mean per buoy
    6. plot_timeseries            -- efficiency curves for 12 buoys
    7. plot_residual_decomp       -- KDE separation of v and u for Epoch 3
    8. plot_triple_frontier       -- NEW: 3-panel frontier scatter (1x3, Epoch 3)
    9. print_degradation_report   -- enhanced terminal report with deficit ranking
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import norm

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
DATA_PATH: str = "dataset2/wec_c5_mock_data_epochs.csv"
TIMESTAMP_COL: str = "PCTimeStamp"
BUOY_COL: str = "Buoy_ID"
TARGET_COL: str = "Energy_Generation_kW"
WPF_COL: str = "Wave_Power_Flux"
EPOCH_COL: str = "Epoch_Marker"

OUTPUT_CAP: float = 350.0       # physical rated capacity of each WEC [kW]
LOG_EPS: float = 1e-6           # guard constant added before log
ROLLING_WINDOW: str = "7D"      # smoothing window for efficiency time series

HEALTHY_BUOYS: List[str] = [f"Boia_{i}" for i in range(1, 9)]
DEGRADED_BUOYS: List[str] = [f"Boia_{i}" for i in range(9, 13)]
ALL_BUOYS: List[str] = HEALTHY_BUOYS + DEGRADED_BUOYS

PLOT_DIR: Path = Path("plots/phase2_SFA/")

PHASE2_CSV_OUT: str = "dataset2/wec_phase2_outputs.csv"

# Colour palette used consistently across all plots
COLOR_HEALTHY: str = "#2471A3"
COLOR_DEGRADED: str = "#C0392B"
COLOR_FRONTIER: str = "#1A5276"


# ===========================================================================
# Section 1 -- Data Loading and Preparation
# ===========================================================================

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Load the raw sensor CSV, compute Wave_Power_Flux if absent, apply
    natural logarithm to both input and output, and sanitise NaN values.

    Log-transformation rationale
    ----------------------------
    The Cobb-Douglas SFA model operates in log-space.  Two domain issues
    must be handled before taking the log:
      (a) zeros: physically possible (calm seas, curtailed buoy)
      (b) values where ln is undefined: not expected but guarded with eps

    A constant eps = 1e-6 is added before log to handle edge cases without
    distorting the distribution of the bulk of valid observations.

    Parameters
    ----------
    csv_path : str
        Path to the raw sensor CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns WPF_COL, 'ln_wpf', 'ln_y' added.
    """
    logger.info("Loading data from: %s", csv_path)
    df: pd.DataFrame = pd.read_csv(csv_path, parse_dates=[TIMESTAMP_COL])
    logger.info("Raw shape: %s", df.shape)

    if WPF_COL not in df.columns:
        logger.info("Column '%s' not found -- computing from Hs and Te", WPF_COL)
        df[WPF_COL] = 0.49 * df["Hs__m"] ** 2 * df["Te__s"]

    df[TARGET_COL] = df[TARGET_COL].clip(upper=OUTPUT_CAP)

    n_before: int = len(df)
    df = df[(df[WPF_COL] > 0) & (df[TARGET_COL] > 0)].copy()
    n_dropped: int = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with non-positive WPF or output", n_dropped)

    for col in [WPF_COL, TARGET_COL]:
        n_nan: int = int(df[col].isna().sum())
        if n_nan > 0:
            df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed %d NaN values in column '%s'", n_nan, col)

    df["ln_wpf"] = np.log(df[WPF_COL] + LOG_EPS)
    df["ln_y"] = np.log(df[TARGET_COL] + LOG_EPS)

    df = df.sort_values([TIMESTAMP_COL, BUOY_COL]).reset_index(drop=True)
    logger.info(
        "Prepared shape: %s | Epochs present: %s",
        df.shape,
        sorted(df[EPOCH_COL].unique()),
    )
    return df


# ===========================================================================
# Section 2 -- MLE Fitting on Epoch 1
# ===========================================================================

def _neg_log_likelihood(
    params: np.ndarray,
    ln_x: np.ndarray,
    ln_y: np.ndarray,
) -> float:
    """
    Negative log-likelihood of the Normal-Half Normal SFA model.

    Internal parameterisation uses unconstrained variables to allow
    gradient-based optimisers to operate without box constraints:
        params[0] : beta_0    (intercept, unconstrained)
        params[1] : beta_1    (elasticity, unconstrained)
        params[2] : log(sigma^2)  --> sigma^2 = exp(params[2]) > 0
        params[3] : log(lambda)   --> lambda  = exp(params[3]) > 0

    Log-likelihood (Aigner, Lovell & Schmidt 1977):
        ln L = N * ln(2)
               - N * ln(sigma)
               - N/2 * ln(2*pi)
               + sum_i [ ln Phi(-lambda * epsilon_i / sigma) ]
               - sum_i [ epsilon_i^2 / (2 * sigma^2) ]

    where epsilon_i = ln_y_i - beta_0 - beta_1 * ln_x_i

    Returns the NEGATIVE log-likelihood (to be minimised).
    """
    beta0: float = params[0]
    beta1: float = params[1]
    sigma2: float = np.exp(params[2])
    lam: float = np.exp(params[3])

    sigma: float = np.sqrt(sigma2)
    epsilon: np.ndarray = ln_y - beta0 - beta1 * ln_x
    n: int = len(epsilon)

    z: np.ndarray = -lam * epsilon / sigma
    log_phi: np.ndarray = norm.logcdf(z)

    log_lik: float = (
        n * np.log(2)
        - n * np.log(sigma)
        - n * 0.5 * np.log(2.0 * np.pi)
        + log_phi.sum()
        - (epsilon ** 2).sum() / (2.0 * sigma2)
    )
    return -log_lik


def fit_sfa_epoch1(df: pd.DataFrame) -> Dict:
    """
    Estimate SFA parameters by Maximum Likelihood using only Epoch 1 data.

    Training on Epoch 1 (Golden Period) guarantees that the estimated
    frontier reflects genuine best-practice operation.  Contaminating the
    training set with Epoch 3 observations (degraded buoys) would shift
    beta_0 downward, making the frontier appear less demanding than it
    truly is and masking part of the degradation signal in the efficiency
    scores.

    Optimisation is performed with L-BFGS-B via scipy.optimize.minimize.
    Multiple restarts are used to mitigate sensitivity to the initial point.

    Parameters
    ----------
    df : pd.DataFrame
        Full prepared DataFrame (function filters to Epoch 1 internally).

    Returns
    -------
    dict with keys:
        beta0, beta1, sigma2, lambda_, sigma_u2, sigma_v2, sigma_star2,
        converged (bool), nll (float)
    """
    df_e1: pd.DataFrame = df[(df[EPOCH_COL] == 1) & (df[TARGET_COL] < 345.0)].copy()
    ln_x: np.ndarray = df_e1["ln_wpf"].values
    ln_y: np.ndarray = df_e1["ln_y"].values

    logger.info(
        "Fitting SFA on Epoch 1 Ramp-up Region: %d observations from %d buoys",
        len(ln_x),
        df_e1[BUOY_COL].nunique(),
    )

    initial_points: List[List[float]] = [
        [3.0, 0.8, np.log(0.5), np.log(1.0)],
        [2.5, 0.9, np.log(0.2), np.log(2.0)],
        [3.5, 0.7, np.log(1.0), np.log(0.5)],
        [3.0, 1.0, np.log(0.1), np.log(3.0)],
    ]

    best_result: Optional[OptimizeResult] = None
    best_nll: float = np.inf

    for x0 in initial_points:
        try:
            res: OptimizeResult = minimize(
                _neg_log_likelihood,
                x0=np.array(x0),
                args=(ln_x, ln_y),
                method="L-BFGS-B",
                options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_result = res
        except Exception as exc:
            logger.warning("Optimisation failed for starting point %s: %s", x0, exc)

    if best_result is None or not best_result.success:
        logger.warning("MLE did not converge cleanly -- check model or data quality")

    beta0: float = best_result.x[0]
    beta1: float = best_result.x[1]
    sigma2: float = np.exp(best_result.x[2])
    lam: float = np.exp(best_result.x[3])

    # Recover structural variances from (sigma^2, lambda):
    #   sigma^2  = sigma_u^2 + sigma_v^2
    #   lambda   = sigma_u   / sigma_v
    # => sigma_u^2 = sigma^2 * lambda^2 / (1 + lambda^2)
    # => sigma_v^2 = sigma^2 * 1        / (1 + lambda^2)
    sigma_u2: float = sigma2 * lam ** 2 / (1.0 + lam ** 2)
    sigma_v2: float = sigma2 * 1.0 / (1.0 + lam ** 2)
    sigma_star2: float = sigma_u2 * sigma_v2 / sigma2

    params: Dict = {
        "beta0": beta0,
        "beta1": beta1,
        "sigma2": sigma2,
        "lambda_": lam,
        "sigma_u2": sigma_u2,
        "sigma_v2": sigma_v2,
        "sigma_star2": sigma_star2,
        "converged": best_result.success,
        "nll": best_nll,
    }

    logger.info("MLE results (Epoch 1 frontier):")
    logger.info("  beta_0    = %+.6f", beta0)
    logger.info("  beta_1    = %+.6f  (output elasticity)", beta1)
    logger.info("  lambda    = %.6f   (sigma_u / sigma_v)", lam)
    logger.info("  sigma^2   = %.6f   (total error variance)", sigma2)
    logger.info("  sigma_u^2 = %.6f   (inefficiency variance)", sigma_u2)
    logger.info("  sigma_v^2 = %.6f   (noise variance)", sigma_v2)
    logger.info("  converged = %s | NLL = %.4f", best_result.success, best_nll)

    return params


# ===========================================================================
# Section 3 -- Efficiency Scoring (Battese-Coelli 1988)
# ===========================================================================

def score_efficiency(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Compute technical efficiency TE_i = E[exp(-u_i) | epsilon_i] for every
    observation in the full dataset using the Battese & Coelli (1988)
    closed-form estimator.

    Derivation
    ----------
    Given the conditional distribution  u_i | epsilon_i ~ TN(mu_star_i, sigma_star^2):

        mu_star_i    = -epsilon_i * sigma_u^2 / sigma^2
        sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2

    The conditional expectation of exp(-u_i) is:

        TE_i = exp(-mu_star_i + sigma_star^2 / 2)
               * Phi(mu_star_i / sigma_star - sigma_star)
               / Phi(mu_star_i / sigma_star)

    Values are clipped to [0, 1].  Values slightly above 1 can arise from
    numerical precision at the efficient frontier (epsilon ~ 0).

    Parameters
    ----------
    df     : Prepared DataFrame (all epochs).
    params : Dict from fit_sfa_epoch1.

    Returns
    -------
    pd.DataFrame with added columns:
        epsilon, mu_star, sigma_noise_hat, SFA_Efficiency
    """
    beta0: float = params["beta0"]
    beta1: float = params["beta1"]
    sigma2: float = params["sigma2"]
    sigma_u2: float = params["sigma_u2"]
    sigma_star2: float = params["sigma_star2"]
    sigma_star: float = np.sqrt(sigma_star2)

    epsilon: np.ndarray = (
        df["ln_y"].values - beta0 - beta1 * df["ln_wpf"].values
    )
    mu_star: np.ndarray = -epsilon * sigma_u2 / sigma2

    ratio: np.ndarray = mu_star / sigma_star
    te: np.ndarray = (
        np.exp(-mu_star + sigma_star2 / 2.0)
        * norm.cdf(ratio - sigma_star)
        / np.maximum(norm.cdf(ratio), 1e-15)
    )
    te = np.clip(te, 0.0, 1.0)

    df = df.copy()
    df["epsilon"] = epsilon
    df["mu_star"] = mu_star
    df["sigma_noise_hat"] = epsilon - (-mu_star)
    df["SFA_Efficiency"] = te

    logger.info(
        "Efficiency scoring complete | mean TE = %.4f | min = %.4f | max = %.4f",
        te.mean(),
        te.min(),
        te.max(),
    )

    summary = (
        df.groupby([EPOCH_COL, BUOY_COL])["SFA_Efficiency"]
        .mean()
        .unstack(BUOY_COL)
        .round(4)
    )
    logger.info("Mean SFA efficiency per epoch and buoy:\n%s", summary.to_string())
    return df


# ===========================================================================
# Section 4 -- Generation Deficit (Distance to Frontier, kW)
# ===========================================================================

def compute_generation_deficit(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Back-transform the SFA frontier into the original kW space and compute
    the Generation_Deficit_kW column for all observations.

    The deterministic frontier in original space is:

        Expected_Y_kW = exp(beta_0 + beta_1 * ln(WPF))

    This is the production the WEC SHOULD achieve if operating on the
    best-practice frontier (u = 0, no inefficiency) for its observed
    wave conditions.  The deficit is:

        Generation_Deficit_kW = Expected_Y_kW - Energy_Generation_kW

    A positive deficit indicates the device produced less than the frontier
    predicts for those wave conditions.  The magnitude is expressed in kW
    and represents the mechanical energy loss attributable to technical
    inefficiency (the u_i term), distinct from the symmetric noise v_i.

    Parameters
    ----------
    df     : DataFrame with SFA_Efficiency already computed.
    params : Dict from fit_sfa_epoch1.

    Returns
    -------
    pd.DataFrame with added columns:
        Expected_Y_kW, Generation_Deficit_kW
    """
    beta0: float = params["beta0"]
    beta1: float = params["beta1"]

    expected_y: np.ndarray = np.exp(beta0 + beta1 * np.log(df[WPF_COL].values + LOG_EPS))
    expected_y = np.clip(expected_y, 0.0, OUTPUT_CAP)

    df = df.copy()
    df["Expected_Y_kW"] = expected_y
    df["Generation_Deficit_kW"] = df["Expected_Y_kW"] - df[TARGET_COL]

    total_deficit_mwh: float = df["Generation_Deficit_kW"].clip(lower=0.0).sum() / 2000.0
    logger.info(
        "Generation deficit computed | mean = %.2f kW | total (positive) = %.1f MWh",
        df["Generation_Deficit_kW"].mean(),
        total_deficit_mwh,
    )
    return df


# ===========================================================================
# Section 5 -- Rolling Aggregation
# ===========================================================================

def aggregate_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 7-day rolling mean of SFA_Efficiency per buoy.

    The rolling window smooths the hourly / 30-minute noise arising from
    natural sea-state variability.  A 7-day span captures the weekly tidal
    cycle and typical storm passage duration, making genuine efficiency
    shifts (onset of mechanical degradation) visible against the noise floor.

    Returns
    -------
    pd.DataFrame
        Wide format: index = PCTimeStamp, columns = Buoy_IDs (rolling mean).
    """
    df_pivot: pd.DataFrame = df.pivot_table(
        index=TIMESTAMP_COL,
        columns=BUOY_COL,
        values="SFA_Efficiency",
    )
    df_pivot.columns.name = None
    rolling: pd.DataFrame = df_pivot.rolling(ROLLING_WINDOW, min_periods=1).mean()
    logger.info("7-day rolling means computed, shape: %s", rolling.shape)
    return rolling


# ===========================================================================
# Section 6 -- Visualisation: Time Series
# ===========================================================================

def _epoch_boundaries(df: pd.DataFrame) -> Dict[int, pd.Timestamp]:
    """
    Extract the start timestamp of each epoch for vertical reference lines.
    """
    return {
        int(epoch): df[df[EPOCH_COL] == epoch][TIMESTAMP_COL].min()
        for epoch in sorted(df[EPOCH_COL].unique())
    }


def plot_timeseries(
    rolling: pd.DataFrame,
    epoch_bounds: Dict[int, pd.Timestamp],
    save_path: str,
) -> None:
    """
    Line chart of 7-day rolling SFA efficiency for all 12 buoys.

    Design rationale:
        - Healthy buoys (1-8): shades of blue/teal, thinner lines
        - Degraded buoys (9-12): shades of red/orange, thicker dashed lines
        - Epoch 2 should show a small uniform dip (systematic -15%)
        - Epoch 3 should show a severe isolated drop for buoys 9-12

    Parameters
    ----------
    rolling      : Wide DataFrame of rolling efficiency (output of aggregate_rolling).
    epoch_bounds : Dict mapping epoch number to start timestamp.
    save_path    : Destination file path for the PNG.
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    palette_healthy: List = sns.color_palette("Blues_r", n_colors=len(HEALTHY_BUOYS))
    palette_degraded: List = sns.color_palette("Reds_r", n_colors=len(DEGRADED_BUOYS))

    for i, buoy in enumerate(HEALTHY_BUOYS):
        if buoy in rolling.columns:
            ax.plot(
                rolling.index, rolling[buoy],
                color=palette_healthy[i], linewidth=1.4, alpha=0.85, label=buoy,
            )

    for i, buoy in enumerate(DEGRADED_BUOYS):
        if buoy in rolling.columns:
            ax.plot(
                rolling.index, rolling[buoy],
                color=palette_degraded[i], linewidth=2.2, alpha=0.95,
                label=f"{buoy} (degraded)", linestyle="--",
            )

    epoch_colors: Dict[int, str] = {1: "#555555", 2: "#E67E22", 3: "#C0392B"}
    epoch_labels: Dict[int, str] = {
        1: "Epoch 1\n(Golden Period)",
        2: "Epoch 2\n(-15% global)",
        3: "Epoch 3\n(PTO fault Boias 9-12)",
    }
    for epoch, ts in epoch_bounds.items():
        ax.axvline(ts, color=epoch_colors[epoch], linestyle=":", linewidth=1.4, alpha=0.7)
        ax.text(ts, 0.04, epoch_labels[epoch], fontsize=8, color=epoch_colors[epoch],
                ha="left", va="bottom")

    ax.axhspan(0.0, 0.55, alpha=0.07, color="#C0392B",
               label="Severe degradation zone (<0.55)")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("SFA Technical Efficiency (TE)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(
        "WEC Phase 2 -- SFA Technical Efficiency: 7-Day Rolling Mean\n"
        "Epoch 2: Spectral Spreading Penalty | Epoch 3: isolated PTO fault (Boias 9-12)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Timeseries plot saved to: %s", save_path)


# ===========================================================================
# Section 7 -- Visualisation: Residual Decomposition
# ===========================================================================

def plot_residual_decomposition(df: pd.DataFrame, save_path: str) -> None:
    """
    KDE plot illustrating the SFA error decomposition for Epoch 3.

    The key message:
        - For healthy buoys (1-8), the composite residual epsilon centres
          near zero: v dominates, u is small => high efficiency.
        - For degraded buoys (9-12), epsilon is strongly negative: the
          large u term drags the residual far from zero.

    Two panels:
        Left  -- KDE of epsilon (composite residual) by group
        Right -- KDE of estimated noise v_hat vs inferred u_hat for one
                 representative healthy and one degraded buoy

    Parameters
    ----------
    df        : Full DataFrame with epsilon, mu_star columns.
    save_path : Destination file path for the PNG.
    """
    df_e3: pd.DataFrame = df[df[EPOCH_COL] == 3].copy()
    df_e3["Group"] = df_e3[BUOY_COL].apply(
        lambda b: "Healthy (Boias 1-8)"
        if b in HEALTHY_BUOYS
        else "Degraded (Boias 9-12)"
    )
    df_e3["u_hat"] = df_e3["mu_star"].clip(lower=0)
    
    df_e3["v_hat"] = df_e3["epsilon"] + df_e3["u_hat"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    palette_grp: Dict[str, str] = {
        "Healthy (Boias 1-8)": COLOR_HEALTHY,
        "Degraded (Boias 9-12)": COLOR_DEGRADED,
    }

    ax_left = axes[0]
    for grp, sub in df_e3.groupby("Group"):
        sns.kdeplot(
            sub["epsilon"], ax=ax_left, label=grp,
            color=palette_grp[grp], linewidth=2.2, fill=True, alpha=0.25,
        )
    ax_left.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero residual")
    ax_left.set_xlabel("Composite Residual epsilon = ln(Y) - frontier", fontsize=10)
    ax_left.set_ylabel("Density", fontsize=10)
    ax_left.set_title(
        "Composite Residual Distribution\nEpoch 3 (by operational group)",
        fontweight="bold", fontsize=10,
    )
    ax_left.legend(fontsize=9)
    ax_left.grid(True, alpha=0.25)

    ax_right = axes[1]
    rep_healthy: str = "Boia_1"
    rep_degraded: str = "Boia_9"

    for buoy, color, label, col_name in [
        (rep_healthy,  COLOR_HEALTHY,  f"{rep_healthy} -- noise v (symmetric)",       "v_hat"),
        (rep_degraded, COLOR_DEGRADED, f"{rep_degraded} -- inferred inefficiency u",  "u_hat"),
    ]:
        sub = df_e3[df_e3[BUOY_COL] == buoy]
        sns.kdeplot(
            sub[col_name], ax=ax_right, label=label,
            color=color, linewidth=2.2, fill=True, alpha=0.22,
        )

    ax_right.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_right.set_xlabel("Error component magnitude", fontsize=10)
    ax_right.set_ylabel("Density", fontsize=10)
    ax_right.set_title(
        "SFA Error Decomposition: v vs u\nEpoch 3 (representative buoys)",
        fontweight="bold", fontsize=10,
    )
    ax_right.legend(fontsize=9)
    ax_right.grid(True, alpha=0.25)

    fig.suptitle(
        "SFA Residual Analysis -- Epoch 3: Statistical Separation of Noise and Inefficiency",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residual decomposition plot saved to: %s", save_path)


# ===========================================================================
# Section 8 -- Visualisation: Triple Frontier Scatter (NEW)
# ===========================================================================

def _compute_frontier_curve(
    df_epoch: pd.DataFrame,
    params: Dict,
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the SFA deterministic frontier line and +/- 1 sigma_v band
    over the WPF range of the supplied DataFrame.

    The frontier is back-transformed from log-space:

        Y_frontier = exp(beta_0 + beta_1 * ln(WPF))

    The stochastic band represents the expected symmetric scatter due to
    noise v ~ N(0, sigma_v^2):

        Y_upper = Y_frontier * exp(+sigma_v)
        Y_lower = Y_frontier * exp(-sigma_v)

    Parameters
    ----------
    df_epoch : DataFrame for the epoch of interest (used for WPF range).
    params   : SFA parameter dict.
    n_points : Number of points for the curve.

    Returns
    -------
    Tuple of (wpf_range, frontier_y, lower_band, upper_band) as ndarray.
    """
    beta0: float = params["beta0"]
    beta1: float = params["beta1"]
    sigma_v: float = np.sqrt(params["sigma_v2"])

    wpf_range: np.ndarray = np.linspace(
        df_epoch[WPF_COL].quantile(0.01),
        df_epoch[WPF_COL].quantile(0.99),
        n_points,
    )
    frontier_y: np.ndarray = np.clip(
        np.exp(beta0 + beta1 * np.log(wpf_range + LOG_EPS)),
        0.0, OUTPUT_CAP,
    )
    upper: np.ndarray = np.clip(frontier_y * np.exp(+sigma_v), 0.0, OUTPUT_CAP)
    lower: np.ndarray = np.clip(frontier_y * np.exp(-sigma_v), 0.0, OUTPUT_CAP)

    return wpf_range, frontier_y, lower, upper


def _draw_frontier_overlay(
    ax: plt.Axes,
    wpf_range: np.ndarray,
    frontier_y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    show_legend_label: bool = True,
) -> None:
    """
    Draw the SFA frontier line and stochastic band on a given Axes.

    Parameters
    ----------
    ax               : Matplotlib Axes to draw on.
    wpf_range        : X values for the frontier curve.
    frontier_y       : Y values of the deterministic frontier.
    lower, upper     : Y values of the +/- 1 sigma_v band.
    show_legend_label: Whether to include labels for the legend.
    """
    lbl_frontier = (
        r"SFA Deterministic Frontier: $\hat{Y}=\exp(\hat\beta_0+\hat\beta_1\ln WPF)$"
        if show_legend_label else "_nolegend_"
    )
    lbl_band = (
        r"$\pm1\,\sigma_v$ stochastic band (noise scatter)"
        if show_legend_label else "_nolegend_"
    )
    ax.plot(
        wpf_range, frontier_y,
        color=COLOR_FRONTIER, linewidth=2.2, linestyle="-",
        label=lbl_frontier, zorder=5,
    )
    ax.fill_between(
        wpf_range, lower, upper,
        alpha=0.12, color=COLOR_HEALTHY, label=lbl_band,
    )


def _buoy_color(buoy: str) -> str:
    """Return the canonical colour for a buoy based on its health group."""
    return COLOR_DEGRADED if buoy in DEGRADED_BUOYS else COLOR_HEALTHY


def plot_triple_frontier(
    df: pd.DataFrame,
    params: Dict,
    save_path: str,
    epoch: int = 3,
    scatter_sample_per_buoy: int = 120,
) -> None:
    """
    Produce a 3-panel (1x3) frontier scatter figure for the specified epoch
    with shared X and Y axes to eliminate auto-scaling distortion.

    The three panels present the same SFA frontier in three complementary
    analytical lenses:

    Panel A -- Population View
        A transparent scatter of all 30-minute observations sampled uniformly
        per buoy, overlaid with the SFA frontier and stochastic noise band.
        Purpose: show the full distributional shape of the data cloud.

    Panel B -- Snapshot View (single common timestamp)
        Exactly 12 points: one per buoy at a single representative timestamp
        dynamically identified at the temporal midpoint of the epoch.  The
        timestamp is selected as the closest available common observation to
        the epoch midpoint, guaranteeing all 12 buoys have a data point.
        Purpose: a fair instantaneous comparison where wave conditions are
        identical across all devices.

    Panel C -- Mean Operating Point
        Each buoy is reduced to its centroid (mean WPF, mean output) across
        the entire epoch.  Plotted with large 'X' markers.
        Purpose: isolate the long-run systematic position of each device
        relative to the frontier, free from timestamp-to-timestamp noise.

    Shared axes (sharex, sharey) guarantee that the visual position of a
    point at (WPF=x, Y=y) is identical across all three panels, preventing
    the auto-scaling illusion where Panel B or C might appear to show a
    different frontier alignment due to a different axis range.

    Parameters
    ----------
    df                     : Full DataFrame with SFA_Efficiency and
                             Generation_Deficit_kW columns.
    params                 : SFA parameter dict from fit_sfa_epoch1.
    save_path              : Destination file path for the PNG.
    epoch                  : Epoch number to visualise (default: 3).
    scatter_sample_per_buoy: Max observations sampled per buoy for Panel A.
    """
    logger.info("Building triple frontier scatter for Epoch %d", epoch)

    df_epoch: pd.DataFrame = df[df[EPOCH_COL] == epoch].copy()
    wpf_range, frontier_y, lower, upper = _compute_frontier_curve(df_epoch, params)

    # ------------------------------------------------------------------
    # Panel B: dynamic common snapshot timestamp
    # Locate the temporal midpoint of the epoch, then find the nearest
    # timestamp that exists for ALL 12 buoys simultaneously.
    # ------------------------------------------------------------------
    epoch_start: pd.Timestamp = df_epoch[TIMESTAMP_COL].min()
    epoch_end: pd.Timestamp = df_epoch[TIMESTAMP_COL].max()
    epoch_mid: pd.Timestamp = epoch_start + (epoch_end - epoch_start) / 2

    # Timestamps common to all buoys
    ts_per_buoy: List[set] = [
        set(df_epoch[df_epoch[BUOY_COL] == b][TIMESTAMP_COL].tolist())
        for b in ALL_BUOYS
        if b in df_epoch[BUOY_COL].unique()
    ]
    if ts_per_buoy:
        common_ts: set = ts_per_buoy[0].intersection(*ts_per_buoy[1:])
    else:
        common_ts = set()

    snapshot_ts: Optional[pd.Timestamp] = None
    if common_ts:
        common_series: pd.Series = pd.Series(sorted(common_ts))
        snapshot_ts = common_series.iloc[
            (common_series - epoch_mid).abs().argmin()
        ]
        logger.info("Snapshot timestamp (Panel B): %s", snapshot_ts)
    else:
        logger.warning(
            "No common timestamp found for all buoys in Epoch %d -- Panel B will be empty",
            epoch,
        )

    # ------------------------------------------------------------------
    # Panel C: mean operating point per buoy
    # ------------------------------------------------------------------
    df_mean: pd.DataFrame = (
        df_epoch.groupby(BUOY_COL)[[WPF_COL, TARGET_COL, "Expected_Y_kW", "Generation_Deficit_kW", "SFA_Efficiency"]]
        .mean()
        .reindex(ALL_BUOYS)
        .dropna()
    )

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 7),
        sharex=True,
        sharey=True,
    )

    panel_titles: List[str] = [
        f"A -- Population View\n(sampled 30-min observations, Epoch {epoch})",
        f"B -- Instantaneous Snapshot\n(single common timestamp: {snapshot_ts.strftime('%Y-%m-%d %H:%M') if snapshot_ts else 'N/A'})",
        f"C -- Mean Operating Point\n(per-buoy epoch mean)",
    ]

    # ------------------------------------------------------------------
    # Panel A: sampled scatter of all Epoch observations
    # ------------------------------------------------------------------
    ax_a: plt.Axes = axes[0]

    # Bulletproof sampling approach (immune to pandas version changes)
    sampled_chunks = []
    for _, group_data in df_epoch.groupby(BUOY_COL):
        n_samples = min(len(group_data), scatter_sample_per_buoy)
        sampled_chunks.append(group_data.sample(n=n_samples, random_state=42))
    df_sample_a: pd.DataFrame = pd.concat(sampled_chunks, ignore_index=True)

    for buoy in ALL_BUOYS:
        sub = df_sample_a[df_sample_a[BUOY_COL] == buoy]
        if sub.empty:
            continue
        ax_a.scatter(
            sub[WPF_COL], sub[TARGET_COL],
            color=_buoy_color(buoy), alpha=0.25, s=14,
            edgecolors="none", label=buoy,
        )

    _draw_frontier_overlay(ax_a, wpf_range, frontier_y, lower, upper, show_legend_label=True)
    ax_a.set_title(panel_titles[0], fontsize=9, fontweight="bold")
    ax_a.set_xlabel("Wave Power Flux (WPF) [kW/m]", fontsize=9)
    ax_a.set_ylabel("Energy Generation [kW]", fontsize=9)
    ax_a.tick_params(labelsize=8)
    ax_a.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # Panel B: single-timestamp snapshot (12 large opaque markers)
    # ------------------------------------------------------------------
    ax_b: plt.Axes = axes[1]
    _draw_frontier_overlay(ax_b, wpf_range, frontier_y, lower, upper, show_legend_label=False)

    if snapshot_ts is not None:
        df_snap: pd.DataFrame = df_epoch[df_epoch[TIMESTAMP_COL] == snapshot_ts]
        for buoy in ALL_BUOYS:
            row = df_snap[df_snap[BUOY_COL] == buoy]
            if row.empty:
                continue

            x_val = row[WPF_COL].values[0]
            y_val = row[TARGET_COL].values[0]
            y_exp = row["Expected_Y_kW"].values[0]
            deficit = row["Generation_Deficit_kW"].values[0]

            if buoy in DEGRADED_BUOYS:
                ax_b.vlines(x=x_val, ymin=y_val, ymax=y_exp, color=COLOR_DEGRADED, 
                            linestyle='--', linewidth=1.5, zorder=5)
                # Anotar visualmente a perda em kW
                ax_b.text(x_val + 0.5, (y_val + y_exp) / 2, f"-{deficit:.0f} kW", 
                          color=COLOR_DEGRADED, fontsize=7.5, fontweight="bold", va='center')

            ax_b.scatter(
                x_val, y_val,
                color=_buoy_color(buoy),
                s=120, alpha=0.92,
                edgecolors="white", linewidths=0.8,
                zorder=6, label=buoy,
            )
            
            # Label each point with its buoy index for traceability
            buoy_idx: str = buoy.split("_")[-1]
            ax_b.annotate(
                buoy_idx,
                xy=(x_val, y_val),
                xytext=(3, 4), textcoords="offset points",
                fontsize=6.5, color=_buoy_color(buoy), fontweight="bold",
            )

    ax_b.set_title(panel_titles[1], fontsize=9, fontweight="bold")
    ax_b.set_xlabel("Wave Power Flux (WPF) [kW/m]", fontsize=9)
    ax_b.tick_params(labelsize=8)
    ax_b.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # Panel C: per-buoy mean operating point (X markers)
    # ------------------------------------------------------------------
    ax_c: plt.Axes = axes[2]
    _draw_frontier_overlay(ax_c, wpf_range, frontier_y, lower, upper, show_legend_label=False)

    best_buoy = df_mean["SFA_Efficiency"].idxmax()
    worst_buoy = df_mean["SFA_Efficiency"].idxmin()

    for buoy in df_mean.index:
        row_wpf: float = df_mean.loc[buoy, WPF_COL]
        row_y: float = df_mean.loc[buoy, TARGET_COL]

        ax_c.scatter(
            row_wpf, row_y,
            marker="X", color=_buoy_color(buoy),
            s=160, alpha=0.95,
            edgecolors="white", linewidths=0.8,
            zorder=6, label=buoy,
        )

        if buoy == worst_buoy:
            deficit = df_mean.loc[buoy, "Generation_Deficit_kW"]
            te = df_mean.loc[buoy, "SFA_Efficiency"]
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_DEGRADED, lw=1.2, alpha=0.9)
            ax_c.annotate(
                f"Worst Asset ({buoy})\nTE: {te:.2f} | Deficit: -{deficit:.1f} kW",
                xy=(row_wpf, row_y), xytext=(row_wpf + 2, row_y - 50),
                arrowprops=dict(facecolor=COLOR_DEGRADED, shrink=0.05, width=1.5, headwidth=5),
                fontsize=8, fontweight="bold", color=COLOR_DEGRADED, bbox=bbox_props, zorder=10
            )
            
        elif buoy == best_buoy:
            te = df_mean.loc[buoy, "SFA_Efficiency"]
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_HEALTHY, lw=1.2, alpha=0.9)
            ax_c.annotate(
                f"Best Asset ({buoy})\nTE: {te:.2f}",
                xy=(row_wpf, row_y), xytext=(row_wpf - 18, row_y + 40),
                arrowprops=dict(facecolor=COLOR_HEALTHY, shrink=0.05, width=1.5, headwidth=5),
                fontsize=8, fontweight="bold", color=COLOR_HEALTHY, bbox=bbox_props, zorder=10
            )
        else:
            # Label normal para as restantes boias
            buoy_idx = buoy.split("_")[-1]
            ax_c.annotate(
                buoy_idx,
                xy=(row_wpf, row_y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color=_buoy_color(buoy), fontweight="bold",
            )

    ax_c.set_title(panel_titles[2], fontsize=9, fontweight="bold")
    ax_c.set_xlabel("Wave Power Flux (WPF) [kW/m]", fontsize=9)
    ax_c.tick_params(labelsize=8)
    ax_c.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # Shared legend: colour groups only (not individual buoys)
    # ------------------------------------------------------------------
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor=COLOR_HEALTHY,  label="Healthy fleet (Boias 1-8)"),
        Patch(facecolor=COLOR_DEGRADED, label="Degraded fleet (Boias 9-12)"),
        Line2D([0], [0], color=COLOR_FRONTIER, linewidth=2, label="SFA Deterministic Frontier"),
        Patch(facecolor=COLOR_HEALTHY, alpha=0.20, label=r"$\pm1\,\sigma_v$ Stochastic Band"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        f"WEC Phase 2 -- SFA Triple Frontier Analysis (Epoch {epoch})\n"
        "Shared axes eliminate auto-scaling distortion; "
        "healthy buoys scatter around the frontier, degraded buoys fall below it",
        fontsize=11, fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Triple frontier scatter saved to: %s", save_path)


# ===========================================================================
# Section 9 -- Enhanced Degradation Terminal Report
# ===========================================================================

def print_degradation_report(df: pd.DataFrame) -> None:
    """
    Emit a structured Asset Performance Report to the logger.

    The report contains two parts:

    Part 1 -- Cross-epoch summary table:
        For each buoy: mean SFA efficiency across Epochs 1, 2, 3, the
        delta from Epoch 1 to Epoch 3, and a degradation status flag.

    Part 2 -- Epoch 3 deficit ranking (NEW):
        For each buoy: mean SFA efficiency in Epoch 3 and the mean
        Generation_Deficit_kW, sorted by deficit in descending order
        (highest energy loss first).  This constitutes the criticality
        ranking for maintenance prioritisation.

    The worst-performing asset is identified at the end of the report.

    Parameters
    ----------
    df : Full DataFrame with SFA_Efficiency and Generation_Deficit_kW.
    """
    separator: str = "=" * 72

    # ------------------------------------------------------------------
    # Part 1: Cross-epoch efficiency summary
    # ------------------------------------------------------------------
    pivot: pd.DataFrame = (
        df.groupby([EPOCH_COL, BUOY_COL])["SFA_Efficiency"]
        .mean()
        .unstack(EPOCH_COL)
        .rename(columns={
            1: "Epoch1_base",
            2: "Epoch 2_Sub_optimal_Spectrum",
            3: "Epoch3_fault",
        })
    )
    pivot["Delta_E1_to_E3"] = pivot["Epoch3_fault"] - pivot["Epoch1_base"]
    pivot["Status"] = pivot["Delta_E1_to_E3"].apply(
        lambda d: "DEGRADATION DETECTED" if d < -0.20 else "normal"
    )

    logger.info("")
    logger.info(separator)
    logger.info("SFA DEGRADATION REPORT -- Phase 2 Summary")
    logger.info(separator)
    logger.info("Part 1 -- Cross-Epoch Efficiency Summary:")
    logger.info(pivot.round(4).to_string())
    logger.info(separator)

    # ------------------------------------------------------------------
    # Part 2: Epoch 3 deficit ranking (new functionality)
    # ------------------------------------------------------------------
    df_e3: pd.DataFrame = df[df[EPOCH_COL] == 3].copy()

    deficit_summary: pd.DataFrame = (
        df_e3.groupby(BUOY_COL)
        .agg(
            Mean_SFA_Efficiency=("SFA_Efficiency", "mean"),
            Mean_Deficit_kW=("Generation_Deficit_kW", "mean"),
        )
        .reindex(ALL_BUOYS)
        .dropna()
        .sort_values("Mean_Deficit_kW", ascending=False)
        .round(4)
    )

    logger.info("Part 2 -- Epoch 3 Asset Criticality Ranking (descending deficit):")
    logger.info(deficit_summary.to_string())
    logger.info(separator)

    # ------------------------------------------------------------------
    # Per-buoy narrative lines
    # ------------------------------------------------------------------
    for buoy, row in deficit_summary.iterrows():
        logger.info(
            "%s | Mean SFA Efficiency = %.4f | Mean Generation Deficit = %.2f kW",
            str(buoy).replace("_", " "),
            row["Mean_SFA_Efficiency"],
            row["Mean_Deficit_kW"],
        )

    # ------------------------------------------------------------------
    # Worst asset conclusion
    # ------------------------------------------------------------------
    worst_buoy: str = str(deficit_summary.index[0])
    worst_deficit: float = float(deficit_summary.iloc[0]["Mean_Deficit_kW"])
    worst_efficiency: float = float(deficit_summary.iloc[0]["Mean_SFA_Efficiency"])

    logger.info(separator)
    logger.info(
        "CONCLUSION -- Worst performing asset: %s",
        worst_buoy.replace("_", " "),
    )
    logger.info(
        "  Mean SFA Technical Efficiency : %.4f (%.1f%% of frontier)",
        worst_efficiency,
        worst_efficiency * 100.0,
    )
    logger.info(
        "  Mean Generation Deficit        : %.2f kW per observation",
        worst_deficit,
    )
    logger.info(
        "Recommendation: prioritise inspection and PTO maintenance of %s.",
        worst_buoy.replace("_", " "),
    )
    logger.info(separator)
    logger.info("")

# ===========================================================================
# Section 10 -- Artefact Export
# ===========================================================================

def export_artefacts(df: pd.DataFrame) -> None:
    """
    Export the data contract for Phase 3 (The Merge).
    Only the essential columns required for the decision engine are saved
    to minimize disk I/O and maintain a clean SCADA-like architecture.
    """
    logger.info("Exporting intermediate artefacts for Phase 3")
    
    export_cols = [
        TIMESTAMP_COL,
        BUOY_COL,
        EPOCH_COL,
        "SFA_Efficiency",
        "Generation_Deficit_kW"
    ]
    
    # Extract only the necessary columns
    df_export = df[export_cols].copy()
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(PHASE2_CSV_OUT), exist_ok=True)
    df_export.to_csv(PHASE2_CSV_OUT, index=False)
    
    logger.info("Phase 2 data contract exported to: %s", PHASE2_CSV_OUT)

# ===========================================================================
# Main Entry Point
# ===========================================================================

def main() -> None:
    """
    Execute the full Phase 2 SFA pipeline end-to-end.

    Stages
    ------
    1. Load and prepare data
    2. Fit SFA on Epoch 1 (Golden Period)
    3. Score all observations (Battese-Coelli 1988)
    4. Compute Generation_Deficit_kW (back-transformed frontier distance)
    5. Rolling aggregation (7-day)
    6. Terminal degradation report
    7. Plot time series
    8. Plot residual decomposition
    9. Plot triple frontier scatter (NEW)
    """
    logger.info("=" * 72)
    logger.info("WEC Phase 2 -- Stochastic Frontier Analysis (SFA)")
    logger.info("=" * 72)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 -- Load
    # ------------------------------------------------------------------
    if os.path.exists(DATA_PATH):
        df: pd.DataFrame = load_and_prepare(DATA_PATH)
    else:
        logger.warning("CSV not found at '%s' -- generating synthetic dataset", DATA_PATH)

        rng = np.random.default_rng(42)
        n_per_buoy: int = 2400

        records: List[pd.DataFrame] = []
        for buoy in ALL_BUOYS:
            for epoch in [1, 2, 3]:
                hs: np.ndarray = rng.uniform(0.8, 5.0, n_per_buoy)
                te: np.ndarray = rng.uniform(5.0, 15.0, n_per_buoy)
                wpf: np.ndarray = 0.49 * hs ** 2 * te

                base_eff: float = (
                    1.0 if epoch == 1
                    else 0.85 if epoch == 2
                    else (0.45 if buoy in DEGRADED_BUOYS else 0.90)
                )

                energy: np.ndarray = (
                    base_eff * 2.8 * wpf + rng.normal(0, 12, n_per_buoy)
                ).clip(1.0, OUTPUT_CAP)

                start: pd.Timestamp = pd.Timestamp(f"2025-0{2 + epoch}-01")
                ts: pd.DatetimeIndex = pd.date_range(start, periods=n_per_buoy, freq="30min")

                records.append(pd.DataFrame({
                    TIMESTAMP_COL: ts,
                    BUOY_COL:      buoy,
                    "Hs__m":       hs,
                    "Te__s":       te,
                    WPF_COL:       wpf,
                    TARGET_COL:    energy,
                    EPOCH_COL:     epoch,
                }))

        df_raw: pd.DataFrame = pd.concat(records, ignore_index=True)
        os.makedirs("dataset2", exist_ok=True)
        df_raw.to_csv(DATA_PATH, index=False)
        logger.info("Synthetic CSV saved to: %s", DATA_PATH)
        df = load_and_prepare(DATA_PATH)

    # ------------------------------------------------------------------
    # Stage 2 -- Fit SFA
    # ------------------------------------------------------------------
    params: Dict = fit_sfa_epoch1(df)

    # ------------------------------------------------------------------
    # Stage 3 -- Score efficiency
    # ------------------------------------------------------------------
    df = score_efficiency(df, params)

    # ------------------------------------------------------------------
    # Stage 4 -- Compute generation deficit
    # ------------------------------------------------------------------
    df = compute_generation_deficit(df, params)

    # ------------------------------------------------------------------
    # Stage 5 -- Rolling aggregation
    # ------------------------------------------------------------------
    rolling: pd.DataFrame = aggregate_rolling(df)

    # ------------------------------------------------------------------
    # Stage 6 -- Terminal report
    # ------------------------------------------------------------------
    print_degradation_report(df)

    # ------------------------------------------------------------------
    # Stage 7 -- Epoch boundaries
    # ------------------------------------------------------------------
    epoch_bounds: Dict[int, pd.Timestamp] = _epoch_boundaries(df)

    # ------------------------------------------------------------------
    # Stage 8 -- Plots
    # ------------------------------------------------------------------
    plot_timeseries(
        rolling,
        epoch_bounds,
        save_path=str(PLOT_DIR / "wec_phase2_SFA_sfa_timeseries.png"),
    )

    plot_residual_decomposition(
        df,
        save_path=str(PLOT_DIR / "wec_phase2_SFA_sfa_residuals.png"),
    )

    plot_triple_frontier(
        df,
        params,
        save_path=str(PLOT_DIR / "wec_phase2_sfa_triple_frontier.png"),
        epoch=3,
    )

    # ------------------------------------------------------------------
    # Stage 9 -- Export Artefacts 
    # ------------------------------------------------------------------
    export_artefacts(df)

    logger.info("Phase 2 SFA complete. Output files written to: %s", PLOT_DIR.resolve())
    logger.info("=" * 72)


if __name__ == "__main__":
    main()