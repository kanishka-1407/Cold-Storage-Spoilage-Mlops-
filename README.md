# MP297 – Cold Storage Spoilage Prediction

An end-to-end MLOps project that predicts crop spoilage in a cooperative cold store in Dehradun: versioned data, tracked experiments, a registered model, a prediction API, a container, CI, Kubernetes files, structured logging, drift monitoring and retraining.

## Project Information

| | |
|---|---|
| **Course** | SCSE3040 – MLOps (Mini Project) |
| **Project name** | Cold Storage Spoilage (Dehradun) |
| **Group ID** | G083 |
| **Problem ID** | MP297 |
| **Client** | Dehradun Cold Chain Cooperative |
| **Task** | Regression: predict `spoilage_pct` |
| **Metric** | MAE (Mean Absolute Error, in percentage points) |

## Team Members

| S. No. | Member Name | Enrolment Number |
|---|---|---|
| 1 | Keshav Kumar | S24CSEU1998 |
| 2 | Kanishka Jain | S24CSEU2023 |
| 3 | Julima Jain | S24CSEU2253 |
| 4 | Shivam Kumar | S24CSEU2025 |

## Problem Description

Farmers near Dehradun store potatoes, onions and apples in a shared cold store, and some lots come out badly spoiled. The cooperative wants to predict the spoilage percentage of each lot from its storage conditions, so that lots at risk can be sold early instead of being thrown away.

**Inputs:** `storage_days`, `temp_set_c`, `humidity_pct`, `power_cuts_hours`, `bag_weight_kg`, `door_openings_per_day`, `crop`, `chamber`
**Target:** `spoilage_pct`

## Data

| File | Rows | Period | Notes |
|---|---|---|---|
| `data/raw/train.csv` | 1000 | Jan–Jun 2026 | Training data, messy. Tracked with DVC. |
| `data/raw/later.csv` | 300 | Jul–Aug 2026 | Clean. Used for drift monitoring and retraining. Tracked with DVC. |
| `data/raw/holdout_features.csv` | 250 | Sep 2026 | No target. Used for the final prediction. |

The CSV files themselves are not stored in Git. Git stores only the `.dvc` pointer files, which contain each file's hash. To work on the project, place the original CSVs from the project zip into `data/raw/` and do not open or save them in Excel.

## Data Quality Findings (Stage 1)

The checks live in `src/mp297/data_checks.py` and were run on the original `train.csv`.

| Problem | Present | Details |
|---|---|---|
| Leaking column | Yes | `spoilage_found_at_unloading`: missing from the holdout file; correlation with the target is 0.999 once the typo rows are excluded. Must not be used as a feature. |
| Unit mix-up | No | Column means are consistent across all three chambers. |
| Sentinel value | Yes | `power_cuts_hours` contains `9999` in 65 rows. |
| Duplicate rows | No | 0 duplicates. |
| Stuck sensor | Yes | `humidity_pct` is stuck at 83 from 2026-04-14 to 2026-05-02. |
| Target typos | Yes | 8 targets are about 10x too large (49.2 to 123.7). |
| Spelling variants | Yes | `crop` has 18 spellings for 3 real categories (potato, onion, apple). |

The full record is in `reports/findings.yaml`.

### Run the checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m mp297.run_checks
```

## Pipeline Overview

```
Raw data → DVC → detect and clean problems → train models
→ MLflow (runs, registry, @champion) → FastAPI /predict
→ Docker image (mp297-api:1.0) → GitHub Actions CI → Kubernetes YAML
→ JSON logging → drift monitoring (PSI) → retrain v2 → promotion rule
→ holdout predictions
```

## Repository Structure

```
README.md
requirements.txt
params.yaml
src/mp297/            data checks, cleaning, training, monitoring code
tests/                pytest tests (including a TestClient test)
data/raw/             DVC pointer files (*.dvc) and holdout_features.csv
app/main.py           FastAPI service: POST /predict, GET /health
models/               champion model loaded by the service
Dockerfile
.dockerignore
.github/workflows/    CI (pytest on every push)
k8s/                  deployment.yaml and service.yaml
reports/findings.yaml data-problem findings, tracking and monitoring results
predictions/          holdout_predictions.csv
mlflow.db             MLflow tracking database
```

## Project Status

| Stage | Work | Status |
|---|---|---|
| 1 | GitHub repo, DVC, README, data checks, `findings.yaml` (data issues) | Done |
| 2 | Cleaning code, v1 training with MLflow, model registry, tests | To do |
| 3 | FastAPI service, Dockerfile, GitHub Actions, Kubernetes files, JSON logging | To do |
| 4 | PSI drift, MAE per chamber, retrain v2, promotion decision, holdout predictions | To do |

## Technologies Used

- Python, Pandas, NumPy, scikit-learn
- DVC (data version control)
- MLflow (experiment tracking and model registry)
- FastAPI and pydantic
- Docker
- GitHub Actions
- Kubernetes
- pytest

## Git Workflow

- One public repository for the group; every member commits from their own GitHub account.
- Work happens on feature branches (for example `data-checks`) and is merged into `main` through pull requests.
- Commit messages are short and descriptive.
