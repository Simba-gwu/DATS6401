# DATS 6401 — Weeks 11 & 12: One Weather Project

**Part A (Week 11):** Streamlit REST client fetches live current/forecast weather for Washington, DC from Open-Meteo, displays metrics and charts, caches valid responses for 10 minutes, and gracefully handles failures.  
**Part B (Week 12):** Train a RandomForestClassifier on **actual historical hourly Open-Meteo weather for Washington, DC**. FastAPI exposes `POST /predict`, a validated request schema, prediction, confidence score, and metadata. Negative/invalid weather values are rejected with HTTP 422.

## Start in Terminal (Mac)

First, extract this ZIP into your existing `DATS6401` folder; the ZIP contains the `week11_12/` folder.

```bash
cd "/Users/simba/Documents/ DATS6401/week11_12"
conda activate dataviz
python -m pip install -r requirements_week11_12.txt
```

## Part A: Streamlit client — Screenshot 1

```bash
python -m streamlit run client_app.py
```

The browser shows current weather in Washington, DC plus two weather forecast charts, the API URL, and its parameters. Take a screenshot of the live data and a chart. To demonstrate graceful error handling, intentionally turn off Wi-Fi or change the endpoint temporarily (then restore it). Avoid stale cached values: click **Refresh live data**.

REST endpoint: `GET https://api.open-meteo.com/v1/forecast`

Example query parameters (all present in the app):

```json
{
  "latitude": 38.9072,
  "longitude": -77.0369,
  "current": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m",
  "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability",
  "timezone": "America/New_York",
  "forecast_days": 2
}
```

### Caching and error handling

`@st.cache_data(ttl=600)` caches successful requests for 10 minutes. `requests.get(..., timeout=15)` and `raise_for_status()` provide failure detection. `st.error()` and `st.stop()` display a readable explanation instead of a traceback.

## Part B: Train real historical-weather model

Stop Streamlit with Control+C or open a new Terminal window. With an internet connection, run:

```bash
cd "/Users/simba/Documents/ DATS6401/week11_12"
conda activate dataviz
python train_model.py
```

This downloads 2024–2025 hourly observations from `https://archive-api.open-meteo.com/v1/archive`, creates an outcome for **rain during the next hour**, trains a Random Forest on the first 80% of time-ordered observations, evaluates it on the last 20%, and saves the trained model to `models/weather_model.joblib`. **Model training must complete successfully before the server can make predictions.** The script prints accuracy, precision, recall, F1, AUC, and the no-rain baseline (real values depend on the actual downloaded data). No model scores have been invented.

The four predictors at current hour `t` are:
- `temperature_2m` (°C)
- `relative_humidity_2m` (%)
- `cloud_cover` (%)
- `wind_speed_10m` (km/h)

The target is `rain` (mm) recorded at hour `t+1` > 0, where historical `rain` refers to rainfall during the preceding hour. Neither current rain nor future variables are used as features. The split is chronological, not random. This is a student demonstration, not a validated operational forecast: uncalibrated Random Forest probabilities are **model scores**, not guaranteed event probabilities. Only DC historical weather was used in training.

### Run the server — Screenshot 2

Keep the server running in a Terminal:

```bash
python -m uvicorn server:app --reload
```

Visit `http://127.0.0.1:8000/docs`. Click `POST /predict` → **Try it out** → enter:

```json
{
  "temperature_2m": 20,
  "relative_humidity_2m": 80,
  "cloud_cover": 85,
  "wind_speed_10m": 10
}
```

Click **Execute**. Screenshot the **HTTP 200** response with `prediction`, `confidence`, `rain_probability`, `no_rain_probability`, `model`, `target` and `training_region`.

### Test an invalid request — Screenshot 3

Repeat in `/docs`, but set humidity to 150%:

```json
{
  "temperature_2m": 20,
  "relative_humidity_2m": 150,
  "cloud_cover": 85,
  "wind_speed_10m": 10
}
```

The server rejects it with **HTTP 422** because relative humidity must be between 0 and 100%. Screenshot the error.

Alternatively with server running:

```bash
python test_api.py
```

### Connect the Streamlit client to the server (optional extra)

Keep the server running, and in a second Terminal run `python -m streamlit run client_app.py`. In the sidebar select **Call local /predict server**, then click **Predict rain in the next hour**. The Streamlit app sends current API measurements to your separate FastAPI model endpoint.

**Cloud deployment caveat:** Streamlit Cloud cannot call `http://127.0.0.1:8000` running on your Mac. The weather-client-only app will still work on Streamlit Cloud; live ML predictions require separate public hosting for FastAPI. For class submission, local `/docs` screenshots demonstrate the server.

## Submission checklist

- Part A: runnable `client_app.py`, endpoint/parameters above, caching + graceful errors
- Part B: `train_model.py`, `server.py`, `models/weather_model.joblib` **after training**, and `/docs` request/response documentation
- Three screenshots: live Streamlit weather client, HTTP 200 prediction, HTTP 422 rejection
- Provide GitHub repository link. Optional separate Streamlit Cloud deployment for `week11_12/client_app.py`.

## Dataset / API attribution

Open-Meteo forecast: <https://open-meteo.com/en/docs>  
Open-Meteo historical weather: <https://open-meteo.com/en/docs/historical-weather-api>  
Open-Meteo and their upstream weather models/reanalysis sources retain their respective attribution/licensing terms. Include attribution when submitting and especially when republishing data.
