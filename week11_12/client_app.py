"""Week 11: Streamlit REST client using live Open-Meteo weather data.

Run: python -m streamlit run client_app.py
Optional: Run FastAPI server in another Terminal and call its /predict endpoint.
"""

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from shared import CITY_NAME, FEATURE_COLUMNS, FORECAST_ENDPOINT, LATITUDE, LONGITUDE

st.set_page_config(page_title="Live Weather & Rain Prediction", page_icon="🌦️", layout="wide")

PARAMETERS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "current": ",".join(FEATURE_COLUMNS),
    "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability",
    "timezone": "America/New_York",
    "forecast_days": 2,
}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather() -> dict:
    """Cache successful responses for ten minutes, handling failures in the UI."""
    response = requests.get(FORECAST_ENDPOINT, params=PARAMETERS, timeout=15)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(str(data.get("reason", "API reported an error")))
    current = data.get("current")
    hourly = data.get("hourly")
    if not isinstance(current, dict) or not isinstance(hourly, dict):
        raise ValueError("Weather response is missing current or hourly observations")
    missing = [key for key in FEATURE_COLUMNS if current.get(key) is None]
    if missing:
        raise ValueError(f"Weather response is missing current features: {', '.join(missing)}")
    if not all(key in hourly for key in ("time", "temperature_2m", "relative_humidity_2m")):
        raise ValueError("Weather response has incomplete hourly chart data")
    return data


st.title("🌦️ Live Weather & Next-Hour Rain Explorer")
st.caption("Week 11: Open-Meteo live REST API client  ·  Week 12: optional local FastAPI model inference")

with st.sidebar:
    st.header("Settings")
    st.write(f"**City:** {CITY_NAME}")
    st.caption("The model is trained on historical weather for this same city.")
    if st.button("Refresh live data"):
        fetch_weather.clear()
    use_prediction_api = st.checkbox("Call local /predict server", value=False)
    server_url = st.text_input("FastAPI base URL", "http://127.0.0.1:8000", disabled=not use_prediction_api)

with st.expander("REST API endpoint and parameters (required for Week 11)"):
    st.code(FORECAST_ENDPOINT)
    st.json(PARAMETERS)
    st.write("HTTP method: **GET**. Cache TTL: **600 seconds**. Failed requests show a message instead of crashing.")

try:
    data = fetch_weather()
except (requests.Timeout, requests.ConnectionError) as exc:
    st.error(f"The weather service could not be reached: {exc}")
    st.info("Check your connection and try Refresh live data. This is the graceful-error demonstration.")
    st.stop()
except (requests.HTTPError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
    st.error(f"Couldn't load valid weather data: {exc}")
    st.info("The dashboard remains available; refresh when the weather API works again.")
    st.stop()

current = data["current"]
st.subheader(f"Current conditions — {CITY_NAME}")
st.caption(f"Observation time: {current.get('time', 'unknown')} ({data.get('timezone', 'America/New_York')})")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", f"{current['temperature_2m']:.1f} °C")
col2.metric("Humidity", f"{current['relative_humidity_2m']:.0f}%")
col3.metric("Cloud cover", f"{current['cloud_cover']:.0f}%")
col4.metric("Wind speed", f"{current['wind_speed_10m']:.1f} km/h")

hourly = pd.DataFrame(data["hourly"])
hourly["time"] = pd.to_datetime(hourly["time"])
hourly = hourly.dropna(subset=["temperature_2m", "relative_humidity_2m"]).head(36)

st.subheader("Next 36 hourly forecast records")
t1, t2 = st.tabs(["Temperature", "Humidity & precipitation probability"])
with t1:
    fig = px.line(hourly, x="time", y="temperature_2m", markers=True,
                  title="Hourly temperature forecast", labels={"time": "Local time", "temperature_2m": "Temperature (°C)"})
    st.plotly_chart(fig, use_container_width=True)
with t2:
    fig = px.line(hourly, x="time", y="relative_humidity_2m", markers=True,
                  title="Hourly relative humidity", labels={"time": "Local time", "relative_humidity_2m": "Humidity (%)"})
    st.plotly_chart(fig, use_container_width=True)
    if "precipitation_probability" in hourly and hourly["precipitation_probability"].notna().any():
        fig_rain = px.bar(hourly, x="time", y="precipitation_probability",
                          title="Open-Meteo precipitation probability (forecast, not our ML model)",
                          labels={"time": "Local time", "precipitation_probability": "Probability (%)"})
        st.plotly_chart(fig_rain, use_container_width=True)

if use_prediction_api:
    st.subheader("Part B — Prediction using our trained model")
    request_data = {name: float(current[name]) for name in FEATURE_COLUMNS}
    st.caption("Sending today's live measurements to our separate local FastAPI server.")
    with st.expander("JSON sent to /predict"):
        st.json(request_data)
    if st.button("Predict rain in the next hour", type="primary"):
        try:
            response = requests.post(f"{server_url.rstrip('/')}/predict", json=request_data, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.success(f"Prediction: {result['prediction']} · Confidence score: {result['confidence']:.1%}")
            st.metric("Model rain score", f"{result['rain_probability']:.1%}")
            st.json(result)
        except (requests.ConnectionError, requests.Timeout):
            st.warning("FastAPI server is not reachable. Start `python -m uvicorn server:app --reload` in another Terminal.")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            st.error(f"Prediction request failed: {exc}")

st.caption("Source: Open-Meteo. Forecast temperatures and rainfall probabilities are provider estimates; the local rain classifier is a separate exploratory ML model, not an official forecast.")
