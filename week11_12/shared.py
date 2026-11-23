"""Shared feature names and units for the weather REST client and prediction server."""

FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
]

# Washington, DC; all training data and live observations use this location.
LATITUDE = 38.9072
LONGITUDE = -77.0369
CITY_NAME = "Washington, DC"

FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
