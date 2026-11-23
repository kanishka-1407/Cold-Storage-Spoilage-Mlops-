"""Stage 2 v1 model training, MLflow tracking, and Model Registry."""
import hashlib
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp297.clean import clean_data


def compute_sha256(filepath: str) -> str:
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()


def build_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    raw_train_path = params["data"]["raw_train"]
    data_sha256 = compute_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    df = clean_data(raw_df, is_train=True, params=params)

    num_features = params["data"]["num_features"]
    cat_features = params["data"]["cat_features"]
    target = params["data"]["target"]

    X = df[num_features + cat_features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=params["training"]["test_size"],
        random_state=params["training"]["random_state"],
    )

    mlflow.set_tracking_uri(params["training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(params["training"]["experiment_name"])

    candidates = {
        "ridge_alpha_1": Ridge(alpha=1.0),
        "rf_200_depth10": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "gbr_100_lr01": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "gbr_200_lr005": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }

    best_mae = float("inf")
    best_run_id = None
    best_model_uri = None
    best_pipeline = None
    best_name = None

    for model_name, estimator in candidates.items():
        with mlflow.start_run(run_name=model_name) as run:
            pipe = Pipeline([
                ("prep", build_preprocessor(num_features, cat_features)),
                ("model", estimator),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))

            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", params["training"]["test_size"])
            mlflow.log_metric("val_mae", mae)
            model_info = mlflow.sklearn.log_model(
                pipe,
                name="model",
                serialization_format="cloudpickle",
            )

            print(f"{model_name} -> val_mae: {mae:.4f} (run_id={run.info.run_id})")

            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id
                best_model_uri = model_info.model_uri
                best_pipeline = pipe
                best_name = model_name

    # Register best v1 model (version 1) and assign @champion alias
    reg_name = params["training"]["registered_model_name"]
    mv = mlflow.register_model(model_uri=best_model_uri, name=reg_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=reg_name,
        alias="champion",
        version=str(mv.version),
    )

    # Save champion artifact to models/model.joblib
    out_path = Path(params["training"]["champion_model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, out_path)

    print("\n=== CHAMPION SUMMARY (for reports/findings.yaml) ===")
    print(f'experiment_name: "{params["training"]["experiment_name"]}"')
    print(f'best_run_id: "{best_run_id}"')
    print(f'registered_model_name: "{reg_name}"')
    print(f"champion_version: {mv.version}")
    print(f"best_model: {best_name} (val_mae={best_mae:.4f})")
    print(f"data_sha256: {data_sha256}")


if __name__ == "__main__":
    main()