"""Prepare public Capital Bikeshare trip records for a fast Streamlit midterm app.

Usage:
    python prepare_midterm_data.py
    python prepare_midterm_data.py --month 202505

Source: https://capitalbikeshare.com/system-data
For reproducibility, the default is April 2025 (202504).
The script outputs aggregate CSVs only, not identifiable user histories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent
SOURCE_BASE = "https://s3.amazonaws.com/capitalbikeshare-data"
COLUMNS = [
    "ride_id", "started_at", "ended_at", "start_station_name", "start_station_id",
    "end_station_name", "end_station_id", "start_lat", "start_lng",
    "end_lat", "end_lng", "member_casual",
]


def download_zip(month: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{month}-capitalbikeshare-tripdata.zip"
    destination = raw_dir / filename
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using downloaded file: {destination}")
        return destination
    url = f"{SOURCE_BASE}/{filename}"
    print(f"Downloading {url}")
    try:
        with requests.get(url, stream=True, timeout=(20, 240)) as response:
            response.raise_for_status()
            with destination.open("wb") as file:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        file.write(block)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    print(f"Downloaded {destination.stat().st_size / 1024**2:.1f} MB")
    return destination


def read_month(zip_path: Path) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        csvs = [n for n in archive.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if not csvs:
            raise ValueError("ZIP does not include any CSV file")
        name = max(csvs, key=lambda n: archive.getinfo(n).file_size)
        print(f"Reading {name}")
        with archive.open(name) as stream:
            df = pd.read_csv(
                stream,
                usecols=lambda col: col in COLUMNS,
                dtype={"start_station_id": "string", "end_station_id": "string"},
                encoding_errors="replace", low_memory=False,
            )
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected CSV schema; missing: {sorted(missing)}")
    return df


def prepare(df: pd.DataFrame, month: str, target_dir: Path) -> dict:
    """Validate trips and write aggregates used by the Streamlit app."""
    original_count = len(df)
    df = df.copy()
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
    df["duration_minutes"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    # The source's file boundaries may follow end time. Keep starts within requested month.
    start_month = pd.Period(year=int(month[:4]), month=int(month[4:]), freq="M")
    valid = (
        df["started_at"].dt.to_period("M").eq(start_month)
        & df["duration_minutes"].ge(1)
        & df["member_casual"].isin(["member", "casual"])
    )
    trips = df.loc[valid].copy()
    if trips.empty:
        raise ValueError("No valid trips after filtering. Verify month and source schema.")
    trips["date"] = trips["started_at"].dt.strftime("%Y-%m-%d")
    trips["hour"] = trips["started_at"].dt.hour

    # Main time series includes valid dockless trips too.
    daily = (trips.groupby(["date", "member_casual"], as_index=False)
             .size().rename(columns={"size": "trips"}))
    hourly = (trips.groupby(["date", "hour", "member_casual"], as_index=False)
              .size().rename(columns={"size": "trips"}))

    # Only station-to-station rides define the station network; do not fabricate
    # stations for dockless e-bikes with missing start/end station IDs.
    for col in ["start_station_id", "end_station_id"]:
        trips[col] = trips[col].str.strip()
    station_mask = (
        trips["start_station_id"].notna()
        & trips["end_station_id"].notna()
        & trips["start_station_id"].ne("")
        & trips["end_station_id"].ne("")
    )
    docked = trips.loc[station_mask].copy()
    routes_daily = (
        docked.groupby(["date", "member_casual", "start_station_id", "end_station_id"], as_index=False)
        .size().rename(columns={"size": "trips", "start_station_id": "start_id", "end_station_id": "end_id"})
    )

    # Build stations from both trip ends, using median reported coords over the month.
    starts = docked[["start_station_id", "start_station_name", "start_lat", "start_lng"]].rename(
        columns={"start_station_id": "station_id", "start_station_name": "station_name", "start_lat": "lat", "start_lng": "lon"}
    )
    ends = docked[["end_station_id", "end_station_name", "end_lat", "end_lng"]].rename(
        columns={"end_station_id": "station_id", "end_station_name": "station_name", "end_lat": "lat", "end_lng": "lon"}
    )
    stations = pd.concat([starts, ends], ignore_index=True)
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.loc[
        stations["lat"].between(38.5, 39.5) & stations["lon"].between(-78.2, -76.3)
    ].copy()
    stations["station_name"] = stations["station_name"].fillna("Unknown station")
    # Some stations have renamed; most common name is a defensible label.
    names = (stations.groupby(["station_id", "station_name"]).size()
             .reset_index(name="n").sort_values(["station_id", "n"], ascending=[True, False])
             .drop_duplicates("station_id")[["station_id", "station_name"]])
    coords = (stations.groupby("station_id", as_index=False)[["lat", "lon"]].median())
    station_info = coords.merge(names, on="station_id", how="left")

    target_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(target_dir / "daily_trips.csv", index=False)
    hourly.to_csv(target_dir / "hourly_trips.csv", index=False)
    routes_daily.to_csv(target_dir / "daily_routes.csv", index=False)
    station_info.to_csv(target_dir / "stations.csv", index=False)
    summary = {
        "month": month,
        "original_rows": int(original_count),
        "valid_trips": int(len(trips)),
        "station_to_station_trips": int(len(docked)),
        "excluded_from_station_network": int(len(trips) - len(docked)),
        "unique_stations_geolocated": int(len(station_info)),
        "date_start": str(daily["date"].min()),
        "date_end": str(daily["date"].max()),
        "source_url": f"{SOURCE_BASE}/{month}-capitalbikeshare-tripdata.zip",
        "notes": "Time charts include valid rides without station IDs. Map and graph only use trips with both station IDs. Dates are local clock time without timezone information."
    }
    (target_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Validated trips: {len(trips):,} of {original_count:,}")
    print(f"Station-to-station rides: {len(docked):,}")
    print(f"Prepared data saved to: {target_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="202504", help="Year and month YYYYMM (default 202504)")
    args = parser.parse_args()
    month = args.month
    if len(month) != 6 or not month.isdigit() or not 1 <= int(month[4:]) <= 12:
        parser.error("--month must be in YYYYMM form, e.g. 202504")
    raw_dir = PROJECT / "data" / "bikeshare_raw"
    target_dir = PROJECT / "data" / "bikeshare_midterm"
    archive = download_zip(month, raw_dir)
    df = read_month(archive)
    prepare(df, month, target_dir)


if __name__ == "__main__":
    main()
