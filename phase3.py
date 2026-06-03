"""
phase3_merge.py
---------------
Phase 3 Decision Engine for Wave Energy Converter (WEC) anomaly detection.

Methodology:
    Merges Phase 1 (XGBoost absolute baseline errors) with Phase 2 (Stochastic
    Frontier Analysis relative efficiency scores) to detect mechanical PTO
    failures while suppressing environmental false positives caused by atypical
    sea states.

State definitions:
    0 - Nominal:                   A=False, B=False
    1 - Environmental False Positive: A=True,  B=False
    2 - Latent Degradation:        A=False, B=True
    3 - Critical Fault:            A=True,  B=True

A 'Maintenance_Alarm' is raised when State 3 density across a 12-hour rolling
window (24 half-hour periods) reaches or exceeds the configured threshold.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# Input file paths
PATH_PHASE1: str = "dataset2/wec_phase1_outputs.csv"
PATH_PHASE2: str = "dataset2/wec_phase2_outputs.csv"

# Output paths
OUTPUT_DIR: Path = Path("plots/phase3_merge")
#OUTPUT_PLOT: Path = OUTPUT_DIR / "wec_phase3_decision_matrix.png"

# Merge key columns
MERGE_KEYS: List[str] = ["PCTimeStamp", "Buoy_ID"]

# Condition A threshold multiplier (absolute error)
CONDITION_A_MULTIPLIER: float = -1

# Condition B threshold (relative efficiency floor)
CONDITION_B_EFFICIENCY_THRESHOLD: float = 0.60


# Epoch to focus on for the terminal report and visualisation
# Epochs to focus on for the terminal report and visualisation
REPORT_EPOCHS: List[int] = [2, 3]

# Buoy groupings for the comparative visualisation
HEALTHY_FLEET: List[str] = [f"Boia_{i}" for i in range(1, 9)]
DEGRADED_FLEET: List[str] = [f"Boia_{i}" for i in range(9, 13)]

# State colour palette (0=nominal, 1=env FP, 2=latent, 3=critical)
STATE_COLORS: List[str] = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]
STATE_LABELS: List[str] = [
    "State 0 - Nominal",
    "State 1 - Env. False Positive",
    "State 2 - Latent Degradation",
    "State 3 - Critical Fault",
]

# Logging format
LOG_FORMAT: str = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger: logging.Logger = logging.getLogger("phase3_merge")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_phase1(path: str) -> pd.DataFrame:
    """Load and validate the Phase 1 XGBoost output CSV.

    Parameters
    ----------
    path:
        Filesystem path to the Phase 1 CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed and typed Phase 1 data.
    """
    logger.info("Loading Phase 1 data from: %s", path)
    df = pd.read_csv(
        path,
        parse_dates=["PCTimeStamp"],
        dtype={
            "Buoy_ID": str,
            "Predicted_Energy_kW": float,
            "Absolute_Residual": float,
            "Is_Absolute_Anomaly": bool,
            "RMSE_test_dynamic": float,
        },
    )
    logger.info("Phase 1 loaded: %d rows, %d columns.", len(df), df.shape[1])
    return df


def load_phase2(path: str) -> pd.DataFrame:
    """Load and validate the Phase 2 SFA output CSV.

    Parameters
    ----------
    path:
        Filesystem path to the Phase 2 CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed and typed Phase 2 data.
    """
    logger.info("Loading Phase 2 data from: %s", path)
    df = pd.read_csv(
        path,
        parse_dates=["PCTimeStamp"],
        dtype={
            "Buoy_ID": str,
            "Epoch_Marker": int,
            "SFA_Efficiency": float,
            "Generation_Deficit_kW": float,
        },
    )
    logger.info("Phase 2 loaded: %d rows, %d columns.", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_phases(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Inner-join Phase 1 and Phase 2 DataFrames on PCTimeStamp and Buoy_ID.

    Parameters
    ----------
    df1:
        Phase 1 DataFrame.
    df2:
        Phase 2 DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame; rows that cannot be matched in both sources are
        discarded.
    """
    logger.info(
        "Merging Phase 1 (%d rows) and Phase 2 (%d rows) on %s.",
        len(df1),
        len(df2),
        MERGE_KEYS,
    )
    merged = pd.merge(df1, df2, on=MERGE_KEYS, how="inner")
    logger.info(
        "Merge complete: %d rows retained (inner join).", len(merged)
    )
    return merged


# ---------------------------------------------------------------------------
# Business logic: instantaneous state scoring
# ---------------------------------------------------------------------------


def compute_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the two diagnostic conditions for every row.

    Condition A (Absolute Error):
        Absolute_Residual < (CONDITION_A_MULTIPLIER * RMSE_test_dynamic)

    Condition B (Relative Efficiency):
        SFA_Efficiency < CONDITION_B_EFFICIENCY_THRESHOLD

    Parameters
    ----------
    df:
        Merged DataFrame containing the required columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with boolean columns 'Cond_A' and 'Cond_B'.
    """
    df = df.copy()
    df["Cond_A"] = (
        df["Absolute_Residual"]
        < (CONDITION_A_MULTIPLIER * df["RMSE_test_dynamic"])
    )
    df["Cond_B"] = df["SFA_Efficiency"] < CONDITION_B_EFFICIENCY_THRESHOLD
    logger.debug(
        "Condition A triggered on %d rows; Condition B on %d rows.",
        df["Cond_A"].sum(),
        df["Cond_B"].sum(),
    )
    return df


def assign_operational_state(df: pd.DataFrame) -> pd.DataFrame:
    """Map (Cond_A, Cond_B) pairs to integer operational states (0-3).

    State Matrix:
        State 0 - Nominal:                   Cond_A=False, Cond_B=False
        State 1 - Environmental False Positive: Cond_A=True,  Cond_B=False
        State 2 - Latent Degradation:        Cond_A=False, Cond_B=True
        State 3 - Critical Fault:            Cond_A=True,  Cond_B=True

    Parameters
    ----------
    df:
        DataFrame containing 'Cond_A' and 'Cond_B' boolean columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with integer column 'Operational_State'.
    """
    df = df.copy()
    conditions = [
        (~df["Cond_A"]) & (~df["Cond_B"]),  # State 0
        df["Cond_A"] & (~df["Cond_B"]),      # State 1
        (~df["Cond_A"]) & df["Cond_B"],      # State 2
        df["Cond_A"] & df["Cond_B"],         # State 3
    ]
    df["Operational_State"] = np.select(conditions, [0, 1, 2, 3], default=0)
    state_counts = df["Operational_State"].value_counts().sort_index()
    for state, count in state_counts.items():
        logger.info(
            "Instantaneous State %d count: %d (%.2f%%)",
            state,
            count,
            100.0 * count / len(df),
        )
    return df



# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------


def generate_terminal_report(df: pd.DataFrame) -> None:
    """Log the O&M dispatch report for Epoch 3 to the terminal.

    For each Buoy_ID present in Epoch 3, reports:
    - Total hours spent with Maintenance_Alarm == True.
    - Whether an O&M vessel dispatch is recommended.

    Dispatch is recommended for any buoy with at least one alarm period.

    Parameters
    ----------
    df:
        Full merged and annotated DataFrame.
    """

    period_duration_hours: float = 0.5

    for epoch in REPORT_EPOCHS:
        epoch_df = df[df["Epoch_Marker"] == epoch].copy()
        if epoch_df.empty:
            logger.warning(
                "No data found for Epoch %d. Terminal report aborted.", epoch
            )
            continue


        logger.info("=" * 70)
        logger.info("WEC FLEET O&M DISPATCH REPORT  --  EPOCH %d", epoch)
        logger.info("=" * 70)

        is_state3 = epoch_df["Operational_State"] == 3
        state3_summary = (
            epoch_df[is_state3].groupby("Buoy_ID").size()
            .mul(period_duration_hours)
            .rename("State3_Hours")
            .reindex(epoch_df["Buoy_ID"].unique(), fill_value=0.0)
            .reset_index()
            .sort_values("Buoy_ID")
        )

        dispatch_required: List[str] = []

        for _, row in state3_summary.iterrows():
            buoy: str = row["Buoy_ID"]
            state3_hours: float = row["State3_Hours"]
            if state3_hours > 0.0:
                dispatch_required.append(buoy)
                logger.info(
                    "  %s | Critical Fault (State 3): %6.1f h | Status: DISPATCH REQUIRED",
                    buoy,
                    state3_hours,
                )
            else:
                logger.info(
                    "  %s | Critical Fault (State 3): %6.1f h | Status: Nominal/Latent - no immediate action",
                    buoy,
                    state3_hours,
                )

        logger.info("-" * 70)
        if dispatch_required:
            logger.info(
                "O&M VESSEL DISPATCH REQUIRED FOR: %s",
                ", ".join(dispatch_required),
            )
        else:
            logger.info("All buoys nominal during Epoch %d. No dispatch required.", epoch)
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _build_state_pivot(
    epoch_df: pd.DataFrame, buoy_list: List[str]
) -> pd.DataFrame:
    """Pivot the operational state data for a subset of buoys.

    Parameters
    ----------
    epoch_df:
        Epoch-filtered DataFrame with 'PCTimeStamp', 'Buoy_ID', and
        'Operational_State'.
    buoy_list:
        Ordered list of Buoy_ID strings to include.

    Returns
    -------
    pd.DataFrame
        Pivot table with timestamps as rows and Buoy_IDs as columns,
        values are integer Operational_States.
    """
    subset = epoch_df[epoch_df["Buoy_ID"].isin(buoy_list)].copy()
    pivot = subset.pivot_table(
        index="PCTimeStamp",
        columns="Buoy_ID",
        values="Operational_State",
        aggfunc="first",
    )
    # Enforce column order and fill sporadic gaps with State 0
    ordered_cols = [b for b in buoy_list if b in pivot.columns]
    pivot = pivot[ordered_cols].fillna(0).astype(int)
    return pivot




def _draw_heatmap_panel(
    ax: plt.Axes,
    state_pivot: pd.DataFrame,
    title: str,
) -> None:
    """Render a single heatmap panel onto an Axes object.

    The heatmap shows Operational_State (0-3) with the alarm overlay drawn
    as semi-transparent hatching on cells where the alarm is active.

    Parameters
    ----------
    ax:
        Matplotlib Axes to draw on.
    state_pivot:
        Pivot table of integer states (rows=time, columns=buoys).
    alarm_pivot:
        Pivot table of boolean alarm flags (same shape).
    title:
        Panel title string.
    """
    cmap = ListedColormap(STATE_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(
        state_pivot.T.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    n_buoys, n_times = state_pivot.T.shape

    ax.set_yticks(range(n_buoys))
    ax.set_yticklabels(state_pivot.columns.tolist(), fontsize=8)

    timestamps = state_pivot.index
    n_ticks = min(12, n_times)
    tick_step = max(1, n_times // n_ticks)
    tick_positions = list(range(0, n_times, tick_step))
    tick_labels = [
        timestamps[i].strftime("%m-%d\n%H:%M") for i in tick_positions
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Timestamp (UTC)", fontsize=9)
    ax.set_ylabel("Buoy ID", fontsize=9)

    return im

def generate_visualisation(df: pd.DataFrame) -> None:
    """Produce and save the Phase 3 decision matrix visualisation.

    Creates a two-panel temporal heatmap for Epoch 3 comparing the healthy
    fleet (Buoys 1-8) against the degraded fleet (Buoys 9-12).

    Parameters
    ----------
    df:
        Full merged and annotated DataFrame.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in REPORT_EPOCHS:
        epoch_df = df[df["Epoch_Marker"] == epoch].copy()
        if epoch_df.empty:
            logger.warning(
                "No Epoch %d data available; visualisation skipped.", epoch
            )
            continue

        logger.info(
            "Generating visualisation for Epoch %d: %d rows.", epoch, len(epoch_df)
        )

        # Build pivot tables for both fleet groups
        healthy_states = _build_state_pivot(epoch_df, HEALTHY_FLEET)
        degraded_states = _build_state_pivot(epoch_df, DEGRADED_FLEET)

        # Figure layout: 2 rows (healthy / degraded), 1 column
        n_healthy = len(healthy_states.columns)
        n_degraded = len(degraded_states.columns)
        row_h_healthy = max(2.0, n_healthy * 0.55)
        row_h_degraded = max(2.0, n_degraded * 0.55)

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(18, row_h_healthy + row_h_degraded + 3),
            gridspec_kw={"height_ratios": [n_healthy, n_degraded]},
            facecolor="white",
        )

        for ax in axes:
            ax.set_facecolor("white")

        _draw_heatmap_panel(
            axes[0],
            healthy_states,
            f"Epoch {epoch} - Healthy Fleet (Buoys 1-8) - Operational State Matrix",
        )
        _draw_heatmap_panel(
            axes[1],
            degraded_states,
            f"Epoch {epoch} - Degraded Fleet (Buoys 9-12) - Operational State Matrix",
        )

        # Shared legend formatted for light background
        legend_elements = [
            Patch(facecolor=c, edgecolor="black", linewidth=1.0, label=lbl)
            for c, lbl in zip(STATE_COLORS, STATE_LABELS)
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=4,
            fontsize=9.5,
            framealpha=0.9,
            facecolor="white",
            edgecolor="black",
            labelcolor="black",
            bbox_to_anchor=(0.5, 0.01),
        )

        # Global title formatting for light background
        fig.suptitle(
            f"WEC Phase 3 Decision Matrix  |  Epoch {epoch}  |  "
            "Healthy vs. Degraded Fleet Comparison",
            fontsize=13,
            fontweight="bold",
            color="black",
            y=0.99,
        )

        # Style axes text and spines for light background
        for ax in axes:
            ax.tick_params(colors="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.0)

        plt.tight_layout(rect=[0, 0.06, 1, 0.97])
        
        # Dynamic output path based on the current epoch
        plot_path = OUTPUT_DIR / f"wec_phase3_decision_matrix_epoch_{epoch}.png"
        
        # Export in high resolution suitable for academic publishing
        fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Visualisation saved to: %s", plot_path)
        
# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> pd.DataFrame:
    """Execute the full Phase 3 decision-engine pipeline.

    Steps:
        1. Load Phase 1 and Phase 2 CSV outputs.
        2. Inner-join on PCTimeStamp and Buoy_ID.
        3. Evaluate diagnostic conditions A and B.
        4. Assign instantaneous operational states (0-3).
        5. Apply the 12-hour rolling alarm filter.
        6. Emit the O&M dispatch terminal report (Epoch 3).
        7. Generate and save the decision-matrix visualisation (Epoch 3).

    Returns
    -------
    pd.DataFrame
        Fully annotated merged DataFrame with all derived columns.
    """
    logger.info("Phase 3 Decision Engine initialised.")

    # --- Step 1: Load inputs ---
    df_phase1 = load_phase1(PATH_PHASE1)
    df_phase2 = load_phase2(PATH_PHASE2)

    # --- Step 2: Merge ---
    df_merged = merge_phases(df_phase1, df_phase2)

    # --- Step 3: Diagnostic conditions ---
    df_conditions = compute_conditions(df_merged)

    # --- Step 4: Operational state assignment ---
    df_states = assign_operational_state(df_conditions)

    # --- Step 5: Terminal report ---
    generate_terminal_report(df_states)

    # --- Step 6: Visualisation ---
    generate_visualisation(df_states)

    logger.info("Phase 3 pipeline complete.")
    return df_states

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    result = run_pipeline()