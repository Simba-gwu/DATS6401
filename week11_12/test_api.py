"""Optional command-line verification of successful and rejected predictions."""
import json
import requests

URL = "http://127.0.0.1:8000/predict"
valid = {
    "temperature_2m": 20.0,
    "relative_humidity_2m": 80.0,
    "cloud_cover": 85.0,
    "wind_speed_10m": 10.0,
}
invalid = {**valid, "relative_humidity_2m": 150.0}

for label, data in [("Valid", valid), ("Invalid", invalid)]:
    try:
        result = requests.post(URL, json=data, timeout=10)
        print(f"\n{label} request: HTTP {result.status_code}")
        print(json.dumps(result.json(), indent=2))
    except requests.RequestException as exc:
        print("Server not reachable:", exc)
        print("Start it with: python -m uvicorn server:app --reload")
        break
