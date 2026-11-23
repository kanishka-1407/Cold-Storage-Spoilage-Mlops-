import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from mp297.clean import clean_data, normalize_crop
from mp297.train import build_preprocessor


def make_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": [1, 2, 3],
        "date": ["2026-03-10", "2026-04-20", "2026-06-01"],
        "storage_days": [15, 30, 45],
        "temp_set_c": [4.0, 2.5, 6.0],
        "humidity_pct": [85.0, 83.0, 88.0],
        "power_cuts_hours": [2.0, 9999.0, 4.0],
        "bag_weight_kg": [50.0, 45.0, 55.0],
        "door_openings_per_day": [3, 5, 2],
        "crop": [" potato ", "ONION", "app."],
        "chamber": ["Chamber 1", "Chamber 2", "Chamber 3"],
        "spoilage_found_at_unloading": [1.0, 8.4, 2.0],
        "spoilage_pct": [5.2, 84.0, 12.1],
    })


def test_normalize_crop_variants():
    assert normalize_crop(" potato ") == "potato"
    assert normalize_crop("POTATO") == "potato"
    assert normalize_crop("oni.") == "onion"
    assert normalize_crop("Apple") == "apple"


def test_clean_data_fixes_leak_sentinel_and_stuck_sensor():
    cleaned = clean_data(make_dirty_df(), is_train=True)
    assert "spoilage_found_at_unloading" not in cleaned.columns
    assert np.isnan(cleaned.loc[1, "power_cuts_hours"])
    assert np.isnan(cleaned.loc[1, "humidity_pct"])
    assert cleaned.loc[0, "humidity_pct"] == 85.0


def test_clean_data_fixes_target_typos_only_when_train():
    train_cleaned = clean_data(make_dirty_df(), is_train=True)
    assert np.isclose(train_cleaned.loc[1, "spoilage_pct"], 8.4)

    infer_cleaned = clean_data(make_dirty_df(), is_train=False)
    assert np.isclose(infer_cleaned.loc[1, "spoilage_pct"], 84.0)


def test_training_pipeline_end_to_end():
    cleaned = clean_data(make_dirty_df(), is_train=True)
    num_cols = [
        "storage_days",
        "temp_set_c",
        "humidity_pct",
        "power_cuts_hours",
        "bag_weight_kg",
        "door_openings_per_day",
    ]
    cat_cols = ["crop", "chamber"]

    pipe = Pipeline([
        ("prep", build_preprocessor(num_cols, cat_cols)),
        ("model", Ridge()),
    ])
    pipe.fit(cleaned[num_cols + cat_cols], cleaned["spoilage_pct"])
    preds = pipe.predict(cleaned[num_cols + cat_cols])

    assert preds.shape == (3,)
    assert not np.isnan(preds).any()