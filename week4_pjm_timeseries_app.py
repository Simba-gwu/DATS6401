import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="PJM Electricity Demand Time Series",
    layout="wide"
)

st.title("PJM Hourly Electricity Demand: Trend, Seasonality, and Uncertainty")

st.caption(
    "Dataset: PJM hourly electricity demand from the Kaggle Hourly Energy Consumption dataset. "
    "Demand is measured in megawatts (MW)."
)

st.write("""
This Streamlit app explores hourly electricity demand over multiple years.
The goal is to visualize long-term trend, repeated seasonal patterns, and uncertainty
around average seasonal demand.
""")


# --------------------------------------------------
# Load data
# --------------------------------------------------
DATA_PATH = Path("data/PJME_hourly.csv")

st.sidebar.header("Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload PJME_hourly.csv if the local file is not found",
    type=["csv"]
)

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.sidebar.success("Loaded data/PJME_hourly.csv")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded CSV file")
else:
    st.error(
        "Could not find data/PJME_hourly.csv. "
        "Download PJME_hourly.csv and place it inside the data folder, or upload it using the sidebar."
    )
    st.stop()


# --------------------------------------------------
# Clean column names and prepare datetime
# --------------------------------------------------
df.columns = df.columns.str.strip()

# Kaggle PJME file usually has columns: Datetime, PJME_MW
if "Datetime" in df.columns:
    date_col = "Datetime"
else:
    date_col = df.columns[0]

if "PJME_MW" in df.columns:
    value_col = "PJME_MW"
else:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        st.error("No numeric demand column found.")
        st.stop()
    value_col = numeric_cols[0]

df = df.rename(columns={
    date_col: "datetime",
    value_col: "demand_mw"
})

df["datetime"] = pd.to_datetime(df["datetime"])
df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")

df = df.dropna(subset=["datetime", "demand_mw"])

# Timestamps are hour-ending ("01:00" = midnight to 1am), so label each hour by its start.
df["datetime"] = df["datetime"] - pd.Timedelta(hours=1)

# Duplicate timestamps can represent the repeated hour when clocks fall back in November.
# They are real electricity readings, so we keep them.
df = df.sort_values("datetime").reset_index(drop=True)


# --------------------------------------------------
# Add time features
# --------------------------------------------------
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["month_name"] = df["datetime"].dt.month_name()
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()
df["date"] = df["datetime"].dt.date

# Identify complete years in the full dataset.
# A complete year should contain all 12 months.
months_per_year = df.groupby("year")["month"].nunique()
complete_years = months_per_year[months_per_year == 12].index.tolist()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Controls")

resolution = st.sidebar.selectbox(
    "Choose time resolution",
    ["Hourly", "Daily", "Weekly", "Monthly", "Yearly"],
    index=1
)

aggregation = st.sidebar.selectbox(
    "Aggregation method",
    ["Mean", "Median", "Max"],
    index=0
)

seasonality_view = st.sidebar.selectbox(
    "Choose seasonality view",
    ["Hour of day", "Day of week", "Month of year"],
    index=0
)

rolling_window = st.sidebar.slider(
    "Rolling average window",
    min_value=2,
    max_value=60,
    value=14,
    step=1
)

date_range = st.sidebar.slider(
    "Choose date range",
    min_value=df["datetime"].min().date(),
    max_value=df["datetime"].max().date(),
    value=(df["datetime"].min().date(), df["datetime"].max().date())
)


# --------------------------------------------------
# Filter by selected date range
# --------------------------------------------------
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

filtered = df[
    (df["datetime"] >= start_date) &
    (df["datetime"] < end_date)
].copy()

if filtered.empty:
    st.warning("No data available for the selected date range.")
    st.stop()


# --------------------------------------------------
# Identify complete years inside selected date range
# --------------------------------------------------
selected_year_stats = filtered.groupby("year").agg(
    month_count=("month", "nunique"),
    hour_count=("demand_mw", "count")
)

# About 90% of a normal year of hourly readings.
selected_complete_years = selected_year_stats[
    (selected_year_stats["month_count"] == 12) &
    (selected_year_stats["hour_count"] >= 7884)
].index.tolist()


# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.subheader("Dataset Preview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{filtered.shape[0]:,}")

with col2:
    st.metric("Start date", str(filtered["datetime"].min().date()))

with col3:
    st.metric("End date", str(filtered["datetime"].max().date()))

with col4:
    st.metric("Complete years selected", len(selected_complete_years))

st.dataframe(filtered.head())


# --------------------------------------------------
# Resample data for trend chart
# --------------------------------------------------
freq_map = {
    "Hourly": "h",
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",
    "Yearly": "YS"
}

freq = freq_map[resolution]

# Minimum number of hourly readings needed before we treat a period as complete enough.
# This avoids counting partial weeks, months, or years as full periods.
min_hours = {
    "Hourly": 1,
    "Daily": 22,
    "Weekly": 151,
    "Monthly": 600,
    "Yearly": 7884
}

stats = filtered.set_index("datetime")["demand_mw"].resample(freq).agg(
    ["mean", "median", "max", "count"]
)

stats = stats[stats["count"] >= min_hours[resolution]]

if stats.empty:
    st.warning("No complete periods are available for this date range and resolution.")
    st.stop()

trend_df = (
    stats[[aggregation.lower()]]
    .rename(columns={aggregation.lower(): "demand_mw"})
    .reset_index()
)

# Start the rolling average only when a full window is available.
trend_df["rolling_average"] = trend_df["demand_mw"].rolling(
    window=rolling_window,
    min_periods=rolling_window
).mean()


# --------------------------------------------------
# Trend chart
# --------------------------------------------------
st.subheader("1. Trend Over Time")

fig_trend = go.Figure()

fig_trend.add_trace(
    go.Scatter(
        x=trend_df["datetime"],
        y=trend_df["demand_mw"],
        mode="lines",
        name=f"{resolution} {aggregation.lower()} demand"
    )
)

fig_trend.add_trace(
    go.Scatter(
        x=trend_df["datetime"],
        y=trend_df["rolling_average"],
        mode="lines",
        name=f"Rolling average ({rolling_window} periods)",
        line=dict(width=4)
    )
)

fig_trend.update_layout(
    title=f"PJM Electricity Demand Over Time ({resolution} Resolution)",
    xaxis_title="Time",
    yaxis_title="Demand (MW)",
    hovermode="x unified"
)

# Interactive date-range brush / range slider
fig_trend.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig_trend, use_container_width=True)

st.write("""
The trend chart shows how electricity demand changes over time. The main line shows
the selected time resolution, while the rolling average smooths the series so the
larger pattern is easier to see. Partial periods are dropped so an incomplete month,
week, or year is not compared as if it were complete.
""")


# --------------------------------------------------
# Bootstrap function
# --------------------------------------------------
def bootstrap_mean_ci(values, n_boot=1000, ci=95, random_state=42):
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)

    return lower, upper


# --------------------------------------------------
# Seasonality setup
# --------------------------------------------------
st.subheader("2. Seasonality With Bootstrap Uncertainty")

if seasonality_view == "Hour of day":
    group_col = "hour"
    x_title = "Hour of day"
    ordered_values = list(range(24))
    label_map = {i: str(i) for i in ordered_values}

elif seasonality_view == "Day of week":
    group_col = "day_of_week"
    x_title = "Day of week"
    ordered_values = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    label_map = {day: day for day in ordered_values}

else:
    group_col = "month"
    x_title = "Month of year"
    ordered_values = list(range(1, 13))
    label_map = {
        i: pd.Timestamp(year=2000, month=i, day=1).month_name()
        for i in ordered_values
    }


# --------------------------------------------------
# Seasonality data with honest bootstrap uncertainty
# --------------------------------------------------
# Use only complete years for the seasonal uncertainty chart.
# This keeps each year comparable.
seasonal_data = filtered[filtered["year"].isin(selected_complete_years)].copy()

if seasonal_data.empty or len(selected_complete_years) < 2:
    st.warning("Pick a date range with at least two complete years to see the seasonal uncertainty band.")
    st.stop()

# One average per (year, group). This is more honest than treating thousands
# of neighboring hourly readings as independent.
yearly_table = (
    seasonal_data
    .groupby(["year", group_col])["demand_mw"]
    .mean()
    .unstack()
)

seasonal_rows = []

for value in ordered_values:
    if value not in yearly_table.columns:
        continue

    # Now each value is one yearly average, not one hourly reading.
    group_values = yearly_table[value].dropna()

    if len(group_values) > 1:
        lower_ci, upper_ci = bootstrap_mean_ci(group_values)

        seasonal_rows.append({
            "season_group": label_map[value],
            "mean_demand": group_values.mean(),
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "n_years": len(group_values)
        })

if not seasonal_rows:
    st.warning("Pick a date range with at least two complete years to see the seasonal uncertainty band.")
    st.stop()

seasonal_df = pd.DataFrame(seasonal_rows)


# --------------------------------------------------
# Seasonality chart
# --------------------------------------------------
fig_season = go.Figure()

# Upper CI invisible line
fig_season.add_trace(
    go.Scatter(
        x=seasonal_df["season_group"],
        y=seasonal_df["upper_ci"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 95% CI"
    )
)

# Lower CI with shaded fill
fig_season.add_trace(
    go.Scatter(
        x=seasonal_df["season_group"],
        y=seasonal_df["lower_ci"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="95% bootstrap confidence interval"
    )
)

# Mean line
fig_season.add_trace(
    go.Scatter(
        x=seasonal_df["season_group"],
        y=seasonal_df["mean_demand"],
        mode="lines+markers",
        name="Average demand"
    )
)

fig_season.update_layout(
    title=f"Average Demand by {seasonality_view}",
    xaxis_title=x_title,
    yaxis_title="Average demand (MW)",
    hovermode="x unified"
)

st.plotly_chart(fig_season, use_container_width=True)

st.write("""
This chart shows seasonality by grouping demand into repeated time units.
The line shows average demand for each seasonal group. The shaded band is a 95%
bootstrap interval from resampling complete years, because neighboring hourly readings
within the same year are correlated and should not be treated as fully independent.
""")


# --------------------------------------------------
# Year-by-year seasonal pattern
# --------------------------------------------------
st.subheader("3. Year-by-Year Seasonal Pattern")

if seasonality_view == "Month of year":
    year_chart_df = (
        seasonal_data
        .groupby(["year", "month"], as_index=False)["demand_mw"]
        .mean()
    )

    fig_year = px.line(
        year_chart_df,
        x="month",
        y="demand_mw",
        color="year",
        markers=True,
        title="Monthly Demand Pattern for Each Complete Year"
    )

    fig_year.update_layout(
        xaxis_title="Month",
        yaxis_title="Average demand (MW)"
    )

elif seasonality_view == "Day of week":
    year_chart_df = (
        seasonal_data
        .groupby(["year", "day_of_week"], as_index=False)["demand_mw"]
        .mean()
    )

    year_chart_df["day_order"] = pd.Categorical(
        year_chart_df["day_of_week"],
        categories=[
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ],
        ordered=True
    )

    year_chart_df = year_chart_df.sort_values("day_order")

    fig_year = px.line(
        year_chart_df,
        x="day_of_week",
        y="demand_mw",
        color="year",
        markers=True,
        title="Weekly Demand Pattern for Each Complete Year"
    )

    fig_year.update_layout(
        xaxis_title="Day of week",
        yaxis_title="Average demand (MW)"
    )

else:
    year_chart_df = (
        seasonal_data
        .groupby(["year", "hour"], as_index=False)["demand_mw"]
        .mean()
    )

    fig_year = px.line(
        year_chart_df,
        x="hour",
        y="demand_mw",
        color="year",
        markers=True,
        title="Daily Demand Pattern for Each Complete Year"
    )

    fig_year.update_layout(
        xaxis_title="Hour of day",
        yaxis_title="Average demand (MW)"
    )

st.plotly_chart(fig_year, use_container_width=True)

st.write("""
This chart separates the seasonal pattern by year. It helps show whether the same
pattern repeats across multiple years. Only complete years are used here, so incomplete
years are not compared against full years.
""")


# --------------------------------------------------
# Monthly distribution
# --------------------------------------------------
st.subheader("4. Monthly Distribution")

month_df = seasonal_data.copy()
month_df["month_name"] = pd.Categorical(
    month_df["month_name"],
    categories=[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ],
    ordered=True
)

fig_box = px.box(
    month_df,
    x="month_name",
    y="demand_mw",
    title="Distribution of Electricity Demand by Month"
)

fig_box.update_layout(
    xaxis_title="Month",
    yaxis_title="Demand (MW)"
)

st.plotly_chart(fig_box, use_container_width=True)

st.write("""
The boxplot shows the spread of demand within each month. This is useful because
seasonal averages alone can hide variation and extreme values.
""")


# --------------------------------------------------
# Dynamic findings for interpretation
# --------------------------------------------------
monthly_avg = seasonal_data.groupby("month")["demand_mw"].mean()
peak_month_num = monthly_avg.idxmax()
peak_month_name = pd.Timestamp(year=2000, month=int(peak_month_num), day=1).month_name()

hourly_avg = seasonal_data.groupby("hour")["demand_mw"].mean()
peak_hour = int(hourly_avg.idxmax())

weekday_data = seasonal_data[
    seasonal_data["day_of_week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
]["demand_mw"].mean()

weekend_data = seasonal_data[
    seasonal_data["day_of_week"].isin(["Saturday", "Sunday"])
]["demand_mw"].mean()

weekday_weekend_gap = weekday_data - weekend_data


# --------------------------------------------------
# Temporal honesty note
# --------------------------------------------------
st.subheader("Temporal Honesty Note")

st.write("""
I made several temporal-honesty choices in this app. First, the raw timestamps are
hour-ending, so I shifted them back one hour; otherwise each hour would be labeled
as the hour after the electricity was used. Second, timestamps use local clock time,
so the repeated November hour during daylight saving time is kept as real data rather
than deleted as a duplicate. Third, the data ends before the final year is complete,
so periods with less than about 90% of their expected hourly readings are dropped
from the trend chart. Fourth, the seasonal uncertainty band resamples complete years
instead of individual hours, because neighboring hourly readings are correlated and
treating them as independent would make the uncertainty band misleadingly narrow.
Finally, the rolling average starts only after a full window is available instead of
averaging fewer points at the edge.
""")


# --------------------------------------------------
# Short interpretation
# --------------------------------------------------
st.subheader("Short Interpretation")

st.write(f"""
The PJM electricity demand series shows both long-term movement and repeated seasonal
patterns. The trend chart shows how demand changes across years, while the rolling
average makes the broader movement easier to see. The seasonality charts show that
demand varies by hour of day, day of week, and month of year.

In the selected complete years, the highest average monthly demand occurs in **{peak_month_name}**.
The average hourly profile reaches its highest level around hour **{peak_hour}**.
The weekday average is about **{weekday_weekend_gap:,.0f} MW** higher than the weekend average,
which suggests that work schedules and human activity affect electricity demand.

The uncertainty band shows that seasonal averages are estimates, not exact truths.
Because electricity demand is correlated over time, the app estimates uncertainty by
resampling complete years rather than individual hourly readings. Overall, the charts
show trend, seasonality, and uncertainty while avoiding misleading comparisons between
complete and incomplete time periods.
""")