"""Data cleaning pipeline for MP297 Cold Storage Spoilage."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_params(path: str = "params.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def normalize_crop(val: object) -> str:
    if pd.isna(val):
        return "unknown"
    cleaned = str(val).strip().lower().rstrip(".")
    if cleaned.startswith("pot"):
        return "potato"
    if cleaned.startswith("oni"):
        return "onion"
    if cleaned.startswith("app"):
        return "apple"
    return cleaned


def clean_data(df: pd.DataFrame, is_train: bool = True, params: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    cfg = (params or load_params()).get("cleaning", {})

    sentinel_val = float(cfg.get("sentinel_power_cuts", 9999.0))
    stuck_val = float(cfg.get("stuck_humidity_val", 83.0))
    stuck_start = pd.Timestamp(cfg.get("stuck_start", "2026-04-14"))
    stuck_end = pd.Timestamp(cfg.get("stuck_end", "2026-05-02"))
    typo_thresh = float(cfg.get("target_typo_threshold", 30.0))

    # 1. Drop leaking column
    if "spoilage_found_at_unloading" in df.columns:
        df = df.drop(columns=["spoilage_found_at_unloading"])

    # 2. Standardize 18 crop spelling variants -> potato, onion, apple
    if "crop" in df.columns:
        df["crop"] = df["crop"].apply(normalize_crop)

    if "chamber" in df.columns:
        df["chamber"] = df["chamber"].astype(str).str.strip()

    # 3. Replace sentinel value 9999 in power_cuts_hours with NaN
    if "power_cuts_hours" in df.columns:
        df["power_cuts_hours"] = pd.to_numeric(df["power_cuts_hours"], errors="coerce").astype(float)
        df.loc[df["power_cuts_hours"] >= sentinel_val, "power_cuts_hours"] = np.nan

    # 4. Mask stuck humidity sensor (83.0 from 2026-04-14 to 2026-05-02)
    if "date" in df.columns and "humidity_pct" in df.columns:
        df["humidity_pct"] = pd.to_numeric(df["humidity_pct"], errors="coerce").astype(float)
        dates = pd.to_datetime(df["date"], errors="coerce")
        stuck_mask = (
            (dates >= stuck_start)
            & (dates <= stuck_end)
            & (df["humidity_pct"] == stuck_val)
        )
        df.loc[stuck_mask, "humidity_pct"] = np.nan

    # 5. Fix 10x target typos (>= 30.0) on labeled data
    if is_train and "spoilage_pct" in df.columns:
        df["spoilage_pct"] = pd.to_numeric(df["spoilage_pct"], errors="coerce").astype(float)
        typo_mask = df["spoilage_pct"] >= typo_thresh
        df.loc[typo_mask, "spoilage_pct"] = df.loc[typo_mask, "spoilage_pct"] / 10.0

    return df