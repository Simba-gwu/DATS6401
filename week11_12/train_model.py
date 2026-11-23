"""Train a real next-hour rain model from Open-Meteo historical weather.

Run once on a computer with an internet connection:
    python train_model.py

Open-Meteo historical hourly `rain` is the preceding-hour rainfall (mm).
For each observation at hour t, the target is whether `rain` in the record
at t+1 was greater than zero. The only predictors are observations at t.
A chronological split prevents fitting on observations after the test period.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from shared import ARCHIVE_ENDPOINT, CITY_NAME, FEATURE_COLUMNS, LATITUDE, LONGITUDE

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "weather_model.joblib"


def download_training_data() -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(FEATURE_COLUMNS + ["rain"]),
        "timezone": "GMT",  # Avoid repeated local-clock timestamps around daylight saving time.
    }
    print("Downloading historical hourly weather from Open-Meteo ...", flush=True)
    response = requests.get(ARCHIVE_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Open-Meteo returned an error"))
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Archive response has no hourly weather values")
    df = pd.DataFrame(hourly)
    required = set(FEATURE_COLUMNS + ["rain", "time"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Archive response lacks columns: {', '.join(sorted(missing))}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    for col in FEATURE_COLUMNS + ["rain"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    next_rain = df["rain"].shift(-1)
    next_time = df["time"].shift(-1)
    one_hour_later = (next_time - df["time"]) == pd.Timedelta(hours=1)
    valid = one_hour_later & next_rain.notna() & df[FEATURE_COLUMNS].notna().all(axis=1)
    X = df.loc[valid, FEATURE_COLUMNS].reset_index(drop=True)
    y = (next_rain.loc[valid] > 0.0).astype(int).reset_index(drop=True)
    timestamps = df.loc[valid, "time"].reset_index(drop=True)
    if len(X) < 500:
        raise ValueError(f"Not enough valid historical observations: {len(X)}")
    return X, y, timestamps


def train() -> dict:
    raw = download_training_data()
    X, y, timestamps = prepare_features(raw)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Train and test partitions must each contain both rain and no-rain examples")

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    rain_idx = list(clf.classes_).index(1)
    probability = clf.predict_proba(X_test)[:, rain_idx]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "precision_rain": round(float(precision_score(y_test, prediction, zero_division=0)), 4),
        "recall_rain": round(float(recall_score(y_test, prediction, zero_division=0)), 4),
        "f1_rain": round(float(f1_score(y_test, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "baseline_accuracy_no_rain": round(float((y_test == 0).mean()), 4),
    }
    metadata = {
        "model": "RandomForestClassifier",
        "trained_on": f"Open-Meteo Historical Weather API: {CITY_NAME}",
        "training_period": f"{START_DATE} through {END_DATE}",
        "train_last_time_utc": timestamps.iloc[split - 1].isoformat(),
        "test_first_time_utc": timestamps.iloc[split].isoformat(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_rain_rate": round(float(y_train.mean()), 4),
        "test_rain_rate": round(float(y_test.mean()), 4),
        "target": "Rainfall > 0 mm during the next hourly interval",
        "feature_units": {
            "temperature_2m": "degrees Celsius",
            "relative_humidity_2m": "percent",
            "cloud_cover": "percent",
            "wind_speed_10m": "km/h",
        },
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "data_source": ARCHIVE_ENDPOINT,
        "note": "Exploratory model; output probabilities are uncalibrated model scores, not certified forecasts.",
    }
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump({"model": clf, "metadata": metadata}, MODEL_PATH)
    print("\nTrained and saved:", MODEL_PATH)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    try:
        train()
    except (requests.RequestException, ValueError, KeyError, OSError) as exc:
        print(f"\nTRAINING ERROR: {exc}", file=sys.stderr)
        print("Check your internet connection and the Open-Meteo archive endpoint, then retry.", file=sys.stderr)
        sys.exit(1)
