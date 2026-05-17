"""
WEC Performance Analysis - Phase 3
=====================================
Stochastic Frontier Analysis (SFA) for technical efficiency scoring and
mechanical degradation detection across 12 Wave Energy Converters (WECs).

Motivation
----------
Phase 2 (DEA) identifies inefficiency deterministically: every deviation
from the empirical frontier is labelled as waste.  In a marine environment
this is a serious flaw because random sea-state variation, sensor noise, and
measurement error are genuinely symmetric and cannot be attributed to the
asset.  SFA addresses this by decomposing the composite residual into two
statistically distinct components:

    epsilon_i = v_i - u_i

where:
    v_i ~ N(0, sigma_v^2)   symmetric noise (waves, sensors)
    u_i ~ |N(0, sigma_u^2)| one-sided technical inefficiency

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
        mu_star_i = -epsilon_i * sigma_u^2 / sigma^2
        sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2

Training strategy: MLE is fitted exclusively on Epoch 1 (Golden Period)
where all buoys are known to operate without mechanical faults.  This
ensures the estimated frontier represents a genuine best-practice production
surface, not a contaminated average.  Epochs 2 and 3 are scored against
this frozen frontier.

Pipeline
--------
    1. load_and_prepare        -- ingest CSV, compute WPF, apply log transform
    2. fit_sfa_epoch1          -- MLE on Epoch 1 only
    3. score_efficiency        -- Battese-Coelli estimator on full dataset
    4. aggregate_rolling       -- 7-day rolling mean per buoy
    5. plot_timeseries         -- efficiency curves for 12 buoys
    6. plot_residual_decomp    -- KDE separation of v and u for Epoch 3
    7. plot_frontier_scatter   -- stochastic frontier vs real data (Epoch 3)
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

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
DATA_PATH       = "dataset2/wec_c5_mock_data_epochs.csv"
TIMESTAMP_COL   = "PCTimeStamp"
BUOY_COL        = "Buoy_ID"
TARGET_COL      = "Energy_Generation_kW"
WPF_COL         = "Wave_Power_Flux"
EPOCH_COL       = "Epoch_Marker"

OUTPUT_CAP      = 350.0       # physical rated capacity of each WEC [kW]
LOG_EPS         = 1e-6        # small constant added before log to avoid domain errors
ROLLING_WINDOW  = "7D"        # smoothing window for efficiency time series

HEALTHY_BUOYS   = [f"Boia_{i}" for i in range(1, 9)]
DEGRADED_BUOYS  = [f"Boia_{i}" for i in range(9, 13)]
ALL_BUOYS       = HEALTHY_BUOYS + DEGRADED_BUOYS

PLOT_DIR        = Path(".")


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
        Cleaned DataFrame with columns WPF_COL, "ln_wpf", "ln_y" added.
    """
    logger.info("Loading data from: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=[TIMESTAMP_COL])
    logger.info("Raw shape: %s", df.shape)

    # Recompute Wave_Power_Flux if the column is absent in the CSV
    if WPF_COL not in df.columns:
        logger.info("Column '%s' not found -- computing from Hs and Te", WPF_COL)
        df[WPF_COL] = 0.49 * df["Hs__m"] ** 2 * df["Te__s"]

    # Cap output at the physical rated power of the device
    df[TARGET_COL] = df[TARGET_COL].clip(upper=OUTPUT_CAP)

    # Drop rows with non-positive WPF or output (cannot take log)
    n_before = len(df)
    df = df[(df[WPF_COL] > 0) & (df[TARGET_COL] > 0)].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with non-positive WPF or output", n_dropped)

    # Impute any remaining NaN with per-column median
    for col in [WPF_COL, TARGET_COL]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed %d NaN values in column '%s'", n_nan, col)

    # Natural log transforms (eps guard included per specification)
    df["ln_wpf"] = np.log(df[WPF_COL] + LOG_EPS)
    df["ln_y"]   = np.log(df[TARGET_COL] + LOG_EPS)

    df = df.sort_values([TIMESTAMP_COL, BUOY_COL]).reset_index(drop=True)
    logger.info("Prepared shape: %s | Epochs present: %s", df.shape, sorted(df[EPOCH_COL].unique()))
    return df


# ===========================================================================
# Section 2 -- MLE Fitting on Epoch 1
# ===========================================================================

def _neg_log_likelihood(params: np.ndarray, ln_x: np.ndarray, ln_y: np.ndarray) -> float:
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
    beta0     = params[0]
    beta1     = params[1]
    sigma2    = np.exp(params[2])
    lam       = np.exp(params[3])

    sigma     = np.sqrt(sigma2)
    epsilon   = ln_y - beta0 - beta1 * ln_x
    n         = len(epsilon)

    # Argument of the standard normal CDF in the likelihood
    z = -lam * epsilon / sigma

    # Log Phi(-lambda * epsilon / sigma): guard against numerical underflow
    log_phi = norm.logcdf(z)

    log_lik = (
        n * np.log(2)
        - n * np.log(sigma)
        - n * 0.5 * np.log(2.0 * np.pi)
        + log_phi.sum()
        - (epsilon ** 2).sum() / (2.0 * sigma2)
    )

    return -log_lik   # minimise the negative


def fit_sfa_epoch1(df: pd.DataFrame) -> dict:
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
        beta0, beta1, sigma2, lambda_, sigma_u2, sigma_v2, sigma_star2
        plus "converged" (bool) and "nll" (final negative log-likelihood)
    """
    df_e1 = df[df[EPOCH_COL] == 1].copy()
    ln_x  = df_e1["ln_wpf"].values
    ln_y  = df_e1["ln_y"].values
    n     = len(ln_x)

    logger.info("Fitting SFA on Epoch 1: %d observations from %d buoys",
                n, df_e1[BUOY_COL].nunique())

    # Multiple starting points: vary sigma^2 and lambda initialisations
    initial_points = [
        [3.0, 0.8, np.log(0.5),  np.log(1.0)],
        [2.5, 0.9, np.log(0.2),  np.log(2.0)],
        [3.5, 0.7, np.log(1.0),  np.log(0.5)],
        [3.0, 1.0, np.log(0.1),  np.log(3.0)],
    ]

    best_result = None
    best_nll    = np.inf

    for x0 in initial_points:
        try:
            res = minimize(
                _neg_log_likelihood,
                x0=np.array(x0),
                args=(ln_x, ln_y),
                method="L-BFGS-B",
                options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res
        except Exception as exc:
            logger.warning("Optimisation failed for starting point %s: %s", x0, exc)

    if best_result is None or not best_result.success:
        logger.warning("MLE did not converge cleanly -- check model or data quality")

    beta0  = best_result.x[0]
    beta1  = best_result.x[1]
    sigma2 = np.exp(best_result.x[2])
    lam    = np.exp(best_result.x[3])

    # Recover structural variances from (sigma^2, lambda)
    # sigma^2  = sigma_u^2 + sigma_v^2
    # lambda   = sigma_u   / sigma_v
    # => sigma_u^2 = sigma^2 * lambda^2 / (1 + lambda^2)
    # => sigma_v^2 = sigma^2 * 1        / (1 + lambda^2)
    sigma_u2    = sigma2 * lam**2 / (1.0 + lam**2)
    sigma_v2    = sigma2 * 1.0    / (1.0 + lam**2)
    sigma_star2 = sigma_u2 * sigma_v2 / sigma2

    params = {
        "beta0":      beta0,
        "beta1":      beta1,
        "sigma2":     sigma2,
        "lambda_":    lam,
        "sigma_u2":   sigma_u2,
        "sigma_v2":   sigma_v2,
        "sigma_star2": sigma_star2,
        "converged":  best_result.success,
        "nll":        best_nll,
    }

    logger.info("MLE results (Epoch 1 frontier):")
    logger.info("  beta_0   = %+.6f", beta0)
    logger.info("  beta_1   = %+.6f  (output elasticity)", beta1)
    logger.info("  lambda   = %.6f   (sigma_u / sigma_v)", lam)
    logger.info("  sigma^2  = %.6f   (total error variance)", sigma2)
    logger.info("  sigma_u^2 = %.6f  (inefficiency variance)", sigma_u2)
    logger.info("  sigma_v^2 = %.6f  (noise variance)", sigma_v2)
    logger.info("  converged = %s | NLL = %.4f", best_result.success, best_nll)

    return params


# ===========================================================================
# Section 3 -- Efficiency Scoring (Battese-Coelli 1988)
# ===========================================================================

def score_efficiency(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute technical efficiency TE_i = E[exp(-u_i) | epsilon_i] for every
    observation in the full dataset using the Battese & Coelli (1988)
    closed-form estimator.

    Derivation
    ----------
    Given the conditional distribution  u_i | epsilon_i ~ TN(mu_star_i, sigma_star^2)
    (truncated normal, truncated at zero):

        mu_star_i  = -epsilon_i * sigma_u^2 / sigma^2
        sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2

    The conditional expectation of exp(-u_i) is:

        TE_i = exp(-mu_star_i + sigma_star^2 / 2)
               * Phi(mu_star_i / sigma_star - sigma_star)
               / Phi(mu_star_i / sigma_star)

    where Phi is the standard normal CDF.

    This formula is numerically stable when mu_star / sigma_star is
    within a reasonable range.  For observations with very large negative
    epsilon (severely underperforming buoys) the ratio Phi(...)/Phi(...)
    may approach numerical zero; the clipping to [1e-6, 1.0] prevents
    log(0) issues in downstream logging while correctly indicating that
    those observations have near-zero efficiency.

    Parameters
    ----------
    df     : prepared DataFrame (all epochs)
    params : dict from fit_sfa_epoch1

    Returns
    -------
    DataFrame with "epsilon", "mu_star", "sigma_noise_hat", "SFA_Efficiency"
    columns added.
    """
    beta0       = params["beta0"]
    beta1       = params["beta1"]
    sigma2      = params["sigma2"]
    sigma_u2    = params["sigma_u2"]
    sigma_star2 = params["sigma_star2"]
    sigma_star  = np.sqrt(sigma_star2)

    # Composite residual: epsilon = ln_y - (beta0 + beta1 * ln_wpf)
    epsilon     = df["ln_y"].values - beta0 - beta1 * df["ln_wpf"].values

    # Conditional mean of u given epsilon (Jondrow et al. 1982)
    mu_star     = -epsilon * sigma_u2 / sigma2

    # Battese-Coelli (1988) efficiency estimator
    ratio       = mu_star / sigma_star
    te = (
        np.exp(-mu_star + sigma_star2 / 2.0)
        * norm.cdf(ratio - sigma_star)
        / np.maximum(norm.cdf(ratio), 1e-15)
    )

    # Clip to [0, 1]: values slightly above 1 can arise from numerical
    # precision at the efficient frontier (epsilon ~ 0)
    te = np.clip(te, 0.0, 1.0)

    df = df.copy()
    df["epsilon"]           = epsilon
    df["mu_star"]           = mu_star
    # Estimated noise component: v = epsilon + u, approximated by residual + E[u|eps]
    df["sigma_noise_hat"]   = epsilon - (-mu_star)   # v_hat = epsilon + u_hat
    df["SFA_Efficiency"]    = te

    logger.info(
        "Efficiency scoring complete | mean TE = %.4f | min = %.4f | max = %.4f",
        te.mean(), te.min(), te.max(),
    )

    # Per-epoch and per-buoy summary
    summary = df.groupby([EPOCH_COL, BUOY_COL])["SFA_Efficiency"].mean().unstack(BUOY_COL)
    epoch_means = summary.round(4)
    logger.info("Mean SFA efficiency per epoch and buoy:\n%s", epoch_means.to_string())

    return df


# ===========================================================================
# Section 4 -- Rolling Aggregation
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
    df_pivot = df.pivot_table(
        index=TIMESTAMP_COL,
        columns=BUOY_COL,
        values="SFA_Efficiency",
    )
    df_pivot.columns.name = None
    rolling = df_pivot.rolling(ROLLING_WINDOW, min_periods=1).mean()
    logger.info("7-day rolling means computed, shape: %s", rolling.shape)
    return rolling


# ===========================================================================
# Section 5 -- Visualisations
# ===========================================================================

def _epoch_boundaries(df: pd.DataFrame) -> dict:
    """
    Extract the start timestamp of each epoch for vertical reference lines.
    """
    bounds = {}
    for epoch in sorted(df[EPOCH_COL].unique()):
        bounds[epoch] = df[df[EPOCH_COL] == epoch][TIMESTAMP_COL].min()
    return bounds


def plot_timeseries(
    rolling: pd.DataFrame,
    epoch_bounds: dict,
    save_path: str,
) -> None:
    """
    Line chart of 7-day rolling SFA efficiency for all 12 buoys.

    Design rationale:
        - Healthy buoys (1-8): shades of blue/teal, thinner lines
        - Degraded buoys (9-12): shades of red/orange, thicker lines
        - Epoch 2 should show a small uniform dip (systematic -15%)
        - Epoch 3 should show a severe isolated drop for buoys 9-12
        - The visual separation makes the SFA fault isolation argument
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    palette_healthy  = sns.color_palette("Blues_r",  n_colors=len(HEALTHY_BUOYS))
    palette_degraded = sns.color_palette("Reds_r",   n_colors=len(DEGRADED_BUOYS))

    for i, buoy in enumerate(HEALTHY_BUOYS):
        if buoy in rolling.columns:
            ax.plot(
                rolling.index, rolling[buoy],
                color=palette_healthy[i], linewidth=1.4,
                alpha=0.85, label=buoy,
            )

    for i, buoy in enumerate(DEGRADED_BUOYS):
        if buoy in rolling.columns:
            ax.plot(
                rolling.index, rolling[buoy],
                color=palette_degraded[i], linewidth=2.2,
                alpha=0.95, label=f"{buoy} (degraded)",
                linestyle="--",
            )

    # Vertical lines at epoch transitions with text labels
    epoch_colors = {1: "#555555", 2: "#E67E22", 3: "#C0392B"}
    epoch_labels = {
        1: "Epoch 1\n(Golden Period)",
        2: "Epoch 2\n(-15% global)",
        3: "Epoch 3\n(PTO fault Boias 9-12)",
    }
    for epoch, ts in epoch_bounds.items():
        ax.axvline(ts, color=epoch_colors[epoch], linestyle=":", linewidth=1.4, alpha=0.7)
        ax.text(ts, 0.04, epoch_labels[epoch], fontsize=8, color=epoch_colors[epoch],
                ha="left", va="bottom", rotation=0)

    ax.axhspan(0.0, 0.55, alpha=0.07, color="#C0392B", label="Severe degradation zone (<0.55)")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("SFA Technical Efficiency (TE)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(
        "WEC Phase 3 -- SFA Technical Efficiency: 7-Day Rolling Mean\n"
        "Epoch 2: uniform environmental shift | Epoch 3: isolated PTO fault (Boias 9-12)",
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


def plot_residual_decomposition(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """
    KDE plot illustrating the SFA error decomposition for Epoch 3.

    The key message:
        - For healthy buoys (1-8), the composite residual epsilon centres
          near zero: v dominates, u is small => high efficiency.
        - For degraded buoys (9-12), epsilon is strongly negative: the
          large u term drags the residual far from zero.
        - The SFA model attributes the healthy scatter to v (symmetric)
          and the downward skew of degraded buoys to u (one-sided).

    Two panels:
        Left  -- KDE of epsilon (composite residual) by group
        Right -- KDE of estimated noise v_hat and inferred u_hat = E[u|eps]
                 separated for one representative healthy and one degraded buoy
    """
    df_e3 = df[df[EPOCH_COL] == 3].copy()
    df_e3["Group"] = df_e3[BUOY_COL].apply(
        lambda b: "Healthy (Boias 1-8)" if b in HEALTHY_BUOYS else "Degraded (Boias 9-12)"
    )
    # u_hat from Jondrow: E[u|eps] = mu_star + sigma_star * phi/Phi
    # Already have mu_star; use a direct approximation via the conditional mean formula
    # u_hat = mu_star + sigma_star * phi(mu_star/sigma_star) / Phi(mu_star/sigma_star)
    sigma_star = np.sqrt(df_e3["mu_star"].std() * 0.1 + 1e-9)   # fallback if not stored
    df_e3["u_hat"] = (-df_e3["mu_star"]).clip(lower=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: composite residual KDE per group
    ax_left = axes[0]
    palette = {"Healthy (Boias 1-8)": "#2471A3", "Degraded (Boias 9-12)": "#C0392B"}
    for grp, sub in df_e3.groupby("Group"):
        sns.kdeplot(
            sub["epsilon"], ax=ax_left, label=grp,
            color=palette[grp], linewidth=2.2, fill=True, alpha=0.25,
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

    # Right panel: inferred v_hat vs u_hat for representative buoys
    ax_right = axes[1]
    rep_healthy  = "Boia_1"
    rep_degraded = "Boia_9"

    for buoy, color, label in [
        (rep_healthy,  "#2471A3", f"{rep_healthy} -- noise v (symmetric)"),
        (rep_degraded, "#C0392B", f"{rep_degraded} -- inferred inefficiency u"),
    ]:
        sub = df_e3[df_e3[BUOY_COL] == buoy]
        col = "epsilon" if buoy == rep_healthy else "u_hat"
        sns.kdeplot(
            sub[col], ax=ax_right, label=label,
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


def plot_frontier_scatter(
    df: pd.DataFrame,
    params: dict,
    save_path: str,
) -> None:
    """
    Scatter plot of Energy_Generation_kW vs Wave_Power_Flux for Epoch 3,
    overlaid with the estimated SFA production frontier.

    The critical visual argument:
        - The SFA frontier is a STOCHASTIC boundary: healthy buoys scatter
          both above and below it (the v term is symmetric).  This is the
          key distinction from DEA, where the frontier is a hard upper envelope.
        - Degraded buoys (9-12) fall systematically BELOW the frontier
          due to their large u (inefficiency) term, not because of bad
          wave conditions.
        - An analyst can visually confirm that the horizontal position
          (Wave_Power_Flux) is similar for degraded and healthy buoys,
          while the vertical position (output) differs sharply -- the fault
          is in the converter, not the sea.

    The frontier line is drawn in the original (non-log) space by
    back-transforming: Y_frontier = exp(beta0 + beta1 * ln(WPF))
    """
    df_e3 = df[df[EPOCH_COL] == 3].copy()
    df_e3["Group"] = df_e3[BUOY_COL].apply(
        lambda b: "Healthy (Boias 1-8)" if b in HEALTHY_BUOYS else "Degraded (Boias 9-12)"
    )

    beta0 = params["beta0"]
    beta1 = params["beta1"]

    # SFA deterministic frontier: exp(beta0 + beta1 * ln(WPF))
    wpf_range  = np.linspace(df_e3[WPF_COL].quantile(0.01),
                             df_e3[WPF_COL].quantile(0.99), 300)
    frontier_y = np.exp(beta0 + beta1 * np.log(wpf_range + LOG_EPS))
    frontier_y = np.clip(frontier_y, 0, OUTPUT_CAP)

    fig, ax = plt.subplots(figsize=(12, 7))

    palette = {
        "Healthy (Boias 1-8)":   "#2471A3",
        "Degraded (Boias 9-12)": "#C0392B",
    }

    # Sample for readability (scatter of all 30-min points is too dense)
    df_sample = df_e3.groupby(BUOY_COL).apply(
        lambda g: g.sample(min(len(g), 120), random_state=42)
    ).reset_index(drop=True)

    for grp, sub in df_sample.groupby("Group"):
        ax.scatter(
            sub[WPF_COL], sub[TARGET_COL],
            color=palette[grp], alpha=0.35, s=18, label=f"Observed -- {grp}",
            edgecolors="none",
        )

    # Deterministic frontier (the estimated mean production function)
    ax.plot(
        wpf_range, frontier_y,
        color="#1A5276", linewidth=2.5, linestyle="-",
        label=r"SFA Deterministic Frontier: $\hat{Y} = \exp(\hat{\beta}_0 + \hat{\beta}_1 \ln WPF)$",
        zorder=5,
    )

    # Stochastic band: +/- 1 sigma_v around the frontier to represent v
    sigma_v = np.sqrt(params["sigma_v2"])
    upper = np.clip(frontier_y * np.exp(+sigma_v), 0, OUTPUT_CAP)
    lower = np.clip(frontier_y * np.exp(-sigma_v), 0, OUTPUT_CAP)
    ax.fill_between(
        wpf_range, lower, upper,
        alpha=0.12, color="#2471A3",
        label=r"$\pm 1\,\sigma_v$ stochastic band (expected noise scatter)",
    )

    ax.set_xlabel("Wave Power Flux (WPF) [kW/m]", fontsize=11)
    ax.set_ylabel("Energy Generation [kW]", fontsize=11)
    ax.set_title(
        "WEC Phase 3 -- SFA Stochastic Frontier vs Observed Output (Epoch 3)\n"
        "Healthy buoys scatter around the frontier (noise v); "
        "degraded buoys deviate downward (inefficiency u)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, OUTPUT_CAP * 1.05)

    # Annotation highlighting the efficiency gap
    ax.annotate(
        "Boias 9-12: systematic\ndownward deviation (large u)",
        xy=(df_e3[df_e3[BUOY_COL] == "Boia_9"][WPF_COL].median(),
            df_e3[df_e3[BUOY_COL] == "Boia_9"][TARGET_COL].median()),
        xytext=(df_e3[WPF_COL].quantile(0.30), OUTPUT_CAP * 0.20),
        fontsize=9, color="#C0392B",
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Frontier scatter plot saved to: %s", save_path)


# ===========================================================================
# Section 6 -- Degradation Terminal Report
# ===========================================================================

def print_degradation_report(df: pd.DataFrame) -> None:
    """
    Print a structured terminal summary of mean SFA efficiency per buoy
    across the three epochs, highlighting anomalous buoys in Epoch 3.
    """
    pivot = (
        df.groupby([EPOCH_COL, BUOY_COL])["SFA_Efficiency"]
        .mean()
        .unstack(EPOCH_COL)
        .rename(columns={1: "Epoch1 (base)", 2: "Epoch2 (-15%)", 3: "Epoch3 (fault)"})
    )
    pivot["Delta E1->E3"] = pivot["Epoch3 (fault)"] - pivot["Epoch1 (base)"]
    pivot["Status"]       = pivot["Delta E1->E3"].apply(
        lambda d: "DEGRADATION DETECTED" if d < -0.20 else "normal"
    )

    logger.info("")
    logger.info("=" * 72)
    logger.info("SFA DEGRADATION REPORT -- Phase 3 Summary")
    logger.info("=" * 72)
    logger.info(pivot.round(4).to_string())
    logger.info("=" * 72)
    logger.info("")


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":

    logger.info("=" * 72)
    logger.info("WEC Phase 3 -- Stochastic Frontier Analysis (SFA)")
    logger.info("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    import os
    if os.path.exists(DATA_PATH):
        df = load_and_prepare(DATA_PATH)
    else:
        # Synthetic fallback: generate a representative dataset so the
        # script can be run end-to-end for validation purposes
        logger.warning("CSV not found at '%s' -- generating synthetic dataset", DATA_PATH)

        rng = np.random.default_rng(42)
        n_per_buoy = 2400   # ~30-min samples over ~50 days per epoch

        records = []
        for buoy in ALL_BUOYS:
            for epoch in [1, 2, 3]:
                n = n_per_buoy
                hs = rng.uniform(0.8, 5.0, n)
                te = rng.uniform(5.0, 15.0, n)
                wpf = 0.49 * hs ** 2 * te

                base_efficiency = 1.0
                if epoch == 2:
                    base_efficiency = 0.85
                elif epoch == 3 and buoy in DEGRADED_BUOYS:
                    base_efficiency = 0.45

                # Energy: physics-based with noise
                energy = (base_efficiency * 2.8 * wpf + rng.normal(0, 12, n)).clip(1.0, OUTPUT_CAP)

                start = pd.Timestamp(f"2025-0{2 + epoch}-01")
                ts    = pd.date_range(start, periods=n, freq="30min")
                records.append(pd.DataFrame({
                    TIMESTAMP_COL: ts,
                    BUOY_COL:      buoy,
                    "Hs__m":       hs,
                    "Te__s":       te,
                    WPF_COL:       wpf,
                    TARGET_COL:    energy,
                    EPOCH_COL:     epoch,
                }))

        df_raw = pd.concat(records, ignore_index=True)
        os.makedirs("dataset2", exist_ok=True)
        df_raw.to_csv(DATA_PATH, index=False)
        logger.info("Synthetic CSV saved to: %s", DATA_PATH)
        df = load_and_prepare(DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Fit SFA on Epoch 1 only
    # ------------------------------------------------------------------
    params = fit_sfa_epoch1(df)

    # ------------------------------------------------------------------
    # 3. Score all observations
    # ------------------------------------------------------------------
    df = score_efficiency(df, params)

    # ------------------------------------------------------------------
    # 4. Rolling aggregation
    # ------------------------------------------------------------------
    rolling = aggregate_rolling(df)

    # ------------------------------------------------------------------
    # 5. Terminal degradation report
    # ------------------------------------------------------------------
    print_degradation_report(df)

    # ------------------------------------------------------------------
    # 6. Epoch boundaries for plot annotations
    # ------------------------------------------------------------------
    epoch_bounds = _epoch_boundaries(df)

    # ------------------------------------------------------------------
    # 7. Generate all three plots
    # ------------------------------------------------------------------
    plot_timeseries(
        rolling,
        epoch_bounds,
        save_path=str(PLOT_DIR / "wec_phase3_sfa_timeseries.png"),
    )

    plot_residual_decomposition(
        df,
        save_path=str(PLOT_DIR / "wec_phase3_sfa_residuals.png"),
    )

    plot_frontier_scatter(
        df,
        params,
        save_path=str(PLOT_DIR / "wec_phase3_sfa_frontier.png"),
    )

    logger.info("Phase 3 complete. Output files:")
    for fname in [
        "wec_phase3_sfa_timeseries.png",
        "wec_phase3_sfa_residuals.png",
        "wec_phase3_sfa_frontier.png",
    ]:
        logger.info("  %s", (PLOT_DIR / fname).resolve())