"""Detectors for the seven kinds of data problem."""
import pandas as pd

TARGET = "spoilage_pct"
GROUP = "chamber"
ID_COLS = ["record_id", "date"]


def load(path):
    return pd.read_csv(path)


def numeric_cols(df):
    return [c for c in df.select_dtypes("number").columns if c != TARGET]


def text_cols(df):
    return [c for c in df.select_dtypes("object").columns if c not in ID_COLS]


# 1. leaking_column
def check_leak(train, holdout):
    missing = [c for c in train.columns if c not in holdout.columns and c != TARGET]
    corr = {}
    for c in train.columns:
        if c == TARGET:
            continue
        x = pd.to_numeric(train[c], errors="coerce")
        y = pd.to_numeric(train[TARGET], errors="coerce")
        if x.notna().sum() > 0:
            corr[c] = round(abs(x.corr(y)), 4)
    corr = dict(sorted(corr.items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 0))
    return {"missing_in_holdout": missing, "abs_corr_with_target": corr}


# 2. unit_mixup (run on data with spelling and sentinels already handled)
def check_unit_mixup(df, group=GROUP):
    means = df.groupby(group)[numeric_cols(df)].mean()
    return {"group_means": means.T.round(2), "max_over_min": (means.max() / means.min()).round(2).to_dict()}


# 3. sentinel_value
def check_sentinels(df, top=3):
    return {c: df[c].value_counts().head(top).to_dict() for c in numeric_cols(df) + [TARGET]}


# 4. duplicate_rows
def check_duplicates(df):
    return {"exact_duplicates": int(df.duplicated().sum()),
            "duplicates_ignoring_record_id": int(df.drop(columns=["record_id"]).duplicated().sum())}


# 5. stuck_sensor
def longest_run(df, col):
    d = df.sort_values(["date", "record_id"]).reset_index(drop=True)
    run_id = (d[col] != d[col].shift()).cumsum()
    rid = run_id.value_counts().idxmax()
    run = d[run_id == rid]
    return {"column": col, "value": float(run[col].iloc[0]), "rows": int(len(run)),
            "start": run["date"].min(), "end": run["date"].max()}


def check_stuck(df):
    runs = [longest_run(df, c) for c in numeric_cols(df)]
    return sorted(runs, key=lambda r: r["rows"], reverse=True)


# 6. target_typos
def check_target_typos(df):
    s = df[TARGET].dropna().sort_values().reset_index(drop=True)
    gaps = s.diff()
    return {"largest_gap": round(float(gaps.max()), 2),
            "gap_starts_after_value": float(s[gaps.idxmax() - 1]),
            "rows_above_gap": int((s >= s[gaps.idxmax()]).sum()),
            "top_15_values": s.tail(15).round(2).tolist()}


# 7. spelling_variants
def check_spelling(df):
    return {c: df[c].value_counts(dropna=False).to_dict() for c in text_cols(df)}