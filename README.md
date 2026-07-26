# Performance Evaluation of Energy Forecasts Service

Technical prototype for energy-production performance evaluation and anomaly detection.

This repository is the current implementation artifact for the broader **Performance Evaluation of Energy Forecasts Service** described in the technical manual/specification. The business goal is to help analysts and operators evaluate forecast or expected-production quality, identify systematic errors, track performance over time, and improve future operational decisions.

In its **current code form**, this repository is an **offline analytical prototype** focused on **Wave Energy Converters (WECs)**. It processes both legacy synthetic data and advanced semi-empirical data anchored on real offshore telemetry. The code evaluates production behaviour through a three-phase pipeline:

1. **Absolute performance modelling** with XGBoost
2. **Relative efficiency analysis** with Stochastic Frontier Analysis (SFA)
3. **Decision fusion** into operational states for maintenance/action support

## Important Scope Note

The specification text refers to a production-grade service for energy forecast evaluation and also mentions PV portfolio use cases. The code in this folder currently implements a **WEC/buoy performance-evaluation prototype**, not an authenticated PV forecasting service.

That means this repository already demonstrates:

- time-aligned analysis over energy-production data
- strict avoidance of temporal data leakage
- handling of real-world telemetry gaps (hard-drop policies)
- residual/error-based performance assessment combined with stochastic frontiers
- systematic underperformance detection
- traceable intermediate artifacts and plots

It does **not** yet provide:

- a REST or gRPC API
- authentication or authorization
- production deployment manifests
- real measured-vs-forecast service endpoints
- automated test coverage or CI packaging

## What The Prototype Does

The prototype analyses a fleet of **12 buoys/WECs** over three operating epochs.

- **Phase 1** learns an expected production baseline from environmental features exclusively (avoiding calendar seasonality bias). It utilizes a metadata-driven "Golden Window" for training and flags large negative residuals (-3 sigma) as absolute anomalies.
- **Phase 2** fits a stochastic production frontier on a healthy reference period. It maps spatial attenuation (wake effects) and separates symmetric environmental noise from one-sided mechanical inefficiency, estimating actual generation deficits.
- **Phase 3** merges both views into a decision matrix that distinguishes environmental false positives (e.g., sub-optimal wave spectra) from verified mechanical PTO degradation.



## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- performance_evaluation.ipynb
|-- dataset1/
|-- dataset2/
`-- plots/
    |-- phase1/
    |-- phase2_SFA/
    `-- phase3_merge/
```

## Pipeline Overview

### Phase 1: Absolute Performance Analysis

Implemented in `performance_evaluation.ipynb`.

Purpose:

- load the fleet dataset (handling real sensor gaps via `IGNORE` metadata)
- engineer wave features while excluding pure temporal variables
- train an `XGBRegressor` on a guaranteed healthy baseline (`Split_Role == TRAIN`)
- predict expected energy generation
- compute residuals against measured generation
- flag absolute anomalies using a strict Statistical Process Control limit (`-3 * RMSE_train`)
- export a data contract for later fusion

Main outputs:

- `[dataset_dir]/wec_phase1_outputs.csv`
- `[dataset_dir]/wec_phase1_xgboost.joblib`
- `plots/phase1/wec_phase1_absolute.png`


### Phase 2: Stochastic Frontier Analysis

Implemented in `performance_evaluation.ipynb`.

Purpose:

- estimate a technical-efficiency frontier from a healthy reference epoch
- separate symmetric noise (environmental shifts) from one-sided inefficiency (asset faults)
- quantify degradation more robustly than absolute residuals alone
- compute rolling efficiency views and generation deficits
- export a second data contract for the decision engine

Main outputs:

- `[dataset_dir]/wec_phase2_outputs.csv`
- `plots/phase2_SFA/wec_phase2_SFA_sfa_timeseries.png`
- `plots/phase2_SFA/wec_phase2_SFA_sfa_residuals.png`
- `plots/phase2_SFA/wec_phase2_sfa_triple_frontier_epoch1.png`
- `plots/phase2_SFA/wec_phase2_sfa_triple_frontier_epoch2.png`
- `plots/phase2_SFA/wec_phase2_sfa_triple_frontier_epoch3.png`


### Phase 3: Decision Engine

Implemented in `performance_evaluation.ipynb`.

**Purpose:**

- merge Phase 1 and Phase 2 outputs
- evaluate two diagnostic conditions:
  - absolute underperformance (Phase 1)
  - relative efficiency degradation (Phase 2)
- assign each observation to an operational state
- generate O&M-oriented reports and decision-matrix plots for Epochs 1, 2, and 3

Operational states:

| State | Meaning |
|---|---|
| 0 | Nominal |
| 1 | Environmental False Positive |
| 2 | Latent Degradation |
| 3 | Critical Fault |

Main outputs:

- `plots/phase3_merge/wec_phase3_decision_matrix_epoch_1.png`
- `plots/phase3_merge/wec_phase3_decision_matrix_epoch_2.png`
- `plots/phase3_merge/wec_phase3_decision_matrix_epoch_3.png`


## Data

The repository supports two distinct data paradigms, managed via dynamic path detection in the code:

**1. Semi-Empirical Dataset (`dataset1/`)**
A highly realistic dataset anchored on real Waverider buoy telemetry (March–June 2026). To simulate a wave farm environment, spatial attenuation factors (wake effects) were applied directly to the significant wave height ($H_s$), alongside the injection of natural oceanic noise. It utilizes strict metadata tagging (`Split_Role`) to handle unrecoverable sensor gaps (`IGNORE`) and enforce robust train/test boundaries without chronological leakage.


**2. Legacy Synthetic Dataset (`dataset2/`)**

The original synthetic dataset used for early methodology validation. It uses a standard chronological 80/20 train/test split.

**Operating Epochs (Scenarios):**

- **Epoch 1**: healthy/golden reference period
- **Epoch 2**: fleet-wide sub-optimal environmental conditions
- **Epoch 3**: Isolated mechanical PTO faults in specific assets (Buoys 9-12).


## Installation

Dependencies are listed in the `requirements.txt` file.

Recommended environment:

- Python 3.10+

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## How To Run

The entire pipeline is unified. Open and execute all cells sequentially in `performance_evaluation.ipynb`.

To toggle between the semi-empirical dataset and the legacy synthetic dataset, simply change the active data directory path variable at the top of each phase.

The pipeline will automatically adapt its internal splitting and export logic based on the provided path, generating all intermediate artifacts and plots for Phase 1, 2, and 3.