# Performance Evaluation of Energy Forecasts Service

Technical prototype for energy-production performance evaluation and anomaly detection.

This repository is the current implementation artifact for the broader **Performance Evaluation of Energy Forecasts Service** described in the technical manual/specification. The business goal is to help analysts and operators evaluate forecast or expected-production quality, identify systematic errors, track performance over time, and improve future operational decisions.

In its **current code form**, this repository is not yet a deployed service API. It is an **offline analytical prototype** focused on **Wave Energy Converters (WECs)** using synthetic buoy data. The code evaluates production behaviour through a three-phase pipeline:

1. **Absolute performance modelling** with XGBoost
2. **Relative efficiency analysis** with Stochastic Frontier Analysis (SFA)
3. **Decision fusion** into operational states for maintenance/action support

## Important Scope Note

The specification text refers to a production-grade service for energy forecast evaluation and also mentions PV portfolio use cases. The code in this folder currently implements a **WEC/buoy performance-evaluation prototype**, not an authenticated PV forecasting service.

That means this repository already demonstrates:

- time-aligned analysis over energy-production data
- residual/error-based performance assessment
- systematic underperformance detection
- traceable intermediate artifacts and plots

It does **not** yet provide:

- a REST or gRPC API
- authentication or authorization
- production deployment manifests
- real measured-vs-forecast service endpoints
- automated test coverage or CI packaging

## What The Prototype Does

The prototype analyses a fleet of **12 buoys/WECs** over three operating epochs using a synthetic SCADA-like dataset.

- **Phase 1** learns an expected production baseline from environmental and temporal features, then flags large negative residuals as absolute anomalies.
- **Phase 2** fits a stochastic production frontier on a healthy reference period and estimates technical efficiency plus generation deficit.
- **Phase 3** merges both views into a decision matrix that distinguishes environmental false positives from likely mechanical degradation.

This makes the repository useful as a research/technical baseline for a future production service concerned with forecast/performance evaluation, explainability, and O&M decision support.

## Repository Structure

```text
.
|-- README.md
|-- performance_evaluation.ipynb
|-- phase1.py
|-- phase1.ipynb
|-- phase2_SFA.py
|-- phase2_SFA.ipynb
|-- phase3.py
|-- phase3.ipynb
|-- dataset2/
|   |-- wec_c5_mock_data_epochs.csv
|   |-- wec_phase1_outputs.csv
|   |-- wec_phase1_xgboost.joblib
|   |-- wec_phase2_outputs.csv
|   `-- create_synt_dataset.py
|-- plots/
|   |-- phase1/
|   |-- phase2_SFA/
|   `-- phase3_merge/
`-- .not_used/
```

## Pipeline Overview

### Phase 1: Absolute Performance Analysis

Implemented in [phase1.py](phase1.py).

Purpose:

- load the synthetic fleet dataset
- engineer wave and temporal features
- train an `XGBRegressor`
- predict expected energy generation
- compute residuals against measured generation
- flag absolute anomalies using a dynamic RMSE-based threshold
- export a data contract for later fusion

Main outputs:

- `dataset2/wec_phase1_outputs.csv`
- `dataset2/wec_phase1_xgboost.joblib`
- `plots/phase1/wec_phase1_absolute.png`

Key exported columns:

- `PCTimeStamp`
- `Buoy_ID`
- `Predicted_Energy_kW`
- `Absolute_Residual`
- `Is_Absolute_Anomaly`
- `RMSE_test_dynamic`

### Phase 2: Stochastic Frontier Analysis

Implemented in [phase2_SFA.py](phase2_SFA.py).

Purpose:

- estimate a technical-efficiency frontier from a healthy reference epoch
- separate symmetric noise from one-sided inefficiency
- quantify degradation more robustly than residuals alone
- compute rolling efficiency views and generation deficits
- export a second data contract for the decision engine

Main outputs:

- `dataset2/wec_phase2_outputs.csv`
- `plots/phase2_SFA/wec_phase2_SFA_sfa_timeseries.png`
- `plots/phase2_SFA/wec_phase2_SFA_sfa_residuals.png`
- `plots/phase2_SFA/wec_phase2_sfa_triple_frontier.png`

Key exported columns:

- `PCTimeStamp`
- `Buoy_ID`
- `Epoch_Marker`
- `SFA_Efficiency`
- `Generation_Deficit_kW`

### Phase 3: Decision Engine

Implemented in [phase3.py](phase3.py).

Purpose:

- merge Phase 1 and Phase 2 outputs
- evaluate two diagnostic conditions:
  - absolute underperformance
  - relative efficiency degradation
- assign each observation to an operational state
- generate O&M-oriented reports and decision-matrix plots

Operational states:

| State | Meaning |
|---|---|
| 0 | Nominal |
| 1 | Environmental False Positive |
| 2 | Latent Degradation |
| 3 | Critical Fault |

Main outputs:

- `plots/phase3_merge/wec_phase3_decision_matrix_epoch_2.png`
- `plots/phase3_merge/wec_phase3_decision_matrix_epoch_3.png`

## Data

The repository includes a synthetic dataset at `dataset2/wec_c5_mock_data_epochs.csv`.

Dataset characteristics:

- 12 wave-energy buoys (`Boia_1` to `Boia_12`)
- 30-minute resolution
- timestamps from January to May 2025
- environmental variables such as `Hs__m` and `Te__s`
- energy output target `Energy_Generation_kW`
- operating-period label `Epoch_Marker`

The epochs are used as scenario markers:

- **Epoch 1**: healthy/golden reference period
- **Epoch 2**: fleet-wide sub-optimal environmental conditions
- **Epoch 3**: separation between healthy and degraded assets

The helper script `dataset2/create_synt_dataset.py` can be used to regenerate mock data, although the repository already contains the dataset needed by the pipeline. If you use the generator as-is, run it with care around the working directory or adjust its output path so the CSV lands in `dataset2/`.

## Installation

No `requirements.txt` is currently included, so dependencies must be installed manually.

Recommended environment:

- Python 3.10+

Suggested packages:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost joblib jupyter
```

## How To Run

Run the three phases in order from the repository root:

```bash
python phase1.py
python phase2_SFA.py
python phase3.py
```

Optional:

- use the notebooks for exploratory analysis and presentation material
- inspect `plots/` after each phase completes
- inspect `dataset2/wec_phase1_outputs.csv` and `dataset2/wec_phase2_outputs.csv` as the intermediate data contracts

## Expected Workflow

1. Phase 1 creates the expected-production baseline and anomaly flags.
2. Phase 2 estimates efficiency and generation deficit.
3. Phase 3 fuses both signals into operational states and visual decision outputs.

This sequencing is important because Phase 3 depends on the CSV artifacts produced by Phases 1 and 2.

## Current Status

This repository should currently be understood as a **research/prototype codebase**, not a finished production service.

Strengths already present:

- clear analytical separation across three phases
- reproducible local execution from raw CSV to plots
- intermediate CSV artifacts that behave like internal data contracts
- interpretable outputs for technical review

Current limitations:

- WEC-specific implementation rather than a generic energy-service abstraction
- synthetic dataset rather than production telemetry and forecast feeds
- no API surface for external consumers
- no configuration layer for multi-tenant/site-level operation
- no test suite, packaging, containerization, or deployment automation
- no monitoring, audit API, or model registry workflow

## How This Maps To The Service Vision

The specification describes a service that evaluates forecast quality by comparing forecasted and measured production over time. This repository partially supports that vision by already providing:

- time-series ingestion and alignment
- model-based expected-production benchmarking
- error/residual analysis
- performance tracking across operating periods
- explainable decision outputs for analysts

To evolve this prototype into the intended production service, typical next steps would be:

1. replace synthetic WEC data with real forecast and measured production feeds
2. externalize configuration for asset/site definitions and evaluation windows
3. wrap the pipeline in an authenticated API
4. add validation, tests, observability, and deployment packaging
5. generalize the domain model to PV/offshore/other energy assets as required

## Intended Audience

This repository is most useful for:

- data scientists validating the methodology
- energy analysts reviewing degradation logic
- software engineers preparing a service implementation
- stakeholders who need a concrete technical baseline from the written specification

## Summary

If you are looking for a **production service**, this repository is the starting point, not the final product.

If you are looking for the **current technical implementation**, this folder contains a coherent three-phase prototype for analysing energy-production behaviour, detecting systematic underperformance, and generating decision-support artifacts for a wave-energy fleet.
