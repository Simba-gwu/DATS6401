"""DATS6401 Midterm: DC bike-share mobility through time, place and connections."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DC Bikeshare Mobility Explorer", page_icon="🚲", layout="wide")
BASE = Path(__file__).resolve().parent / "data" / "bikeshare_midterm"

@st.cache_data(show_spinner=False)
def load_prepared_data():
    daily = pd.read_csv(BASE / "daily_trips.csv", dtype={"member_casual": "string"})
    hourly = pd.read_csv(BASE / "hourly_trips.csv", dtype={"member_casual": "string"})
    routes = pd.read_csv(BASE / "daily_routes.csv", dtype={
        "start_id": "string", "end_id": "string", "member_casual": "string"
    })
    stations = pd.read_csv(BASE / "stations.csv", dtype={"station_id": "string"})
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    for frame in [daily, hourly, routes]:
        frame["date"] = pd.to_datetime(frame["date"])
    return daily, hourly, routes, stations, metadata

st.title("🚲 DC Bikeshare Mobility Explorer")
st.write("**Research question:** When do people ride, where is station activity concentrated, and which stations are most strongly connected?")
st.caption("Three modalities from the same official Capital Bikeshare trip-history source: time series, geographic maps, and station-to-station networks.")

try:
    daily, hourly, routes, stations, meta = load_prepared_data()
except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error("Prepared data is missing or invalid. Please run the download/preprocessing script first.")
    st.code("python prepare_midterm_data.py\npython -m streamlit run midterm_bikeshare_app.py", language="bash")
    st.caption(f"Expected files under: {BASE}")
    with st.expander("Error details"):
        st.exception(error)
    st.stop()

min_date = daily["date"].min().date()
max_date = daily["date"].max().date()
st.sidebar.header("Explore the dataset")
dates = st.sidebar.date_input("Trip start dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(dates, tuple) and len(dates) == 2:
    start_date, end_date = dates
else:
    st.sidebar.info("Select an end date to apply your date range.")
    start_date, end_date = min_date, max_date
rider = st.sidebar.selectbox("Rider type", ["All riders", "Member", "Casual"])
show_type = None if rider == "All riders" else rider.lower()

def selection(data):
    out = data.loc[data["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if show_type is not None:
        out = out.loc[out["member_casual"] == show_type]
    return out

daily_sel = selection(daily)
hourly_sel = selection(hourly)
routes_sel = selection(routes)

# Aggregate station-to-station trips for the selected dates and rider type.
route_counts = (routes_sel.groupby(["start_id", "end_id"], as_index=False)["trips"].sum())
if not route_counts.empty:
    starts = route_counts.groupby("start_id", as_index=False)["trips"].sum().rename(
        columns={"start_id": "station_id", "trips": "departures"})
    ends = route_counts.groupby("end_id", as_index=False)["trips"].sum().rename(
        columns={"end_id": "station_id", "trips": "arrivals"})
    activity = starts.merge(ends, on="station_id", how="outer").fillna(0)
    activity["station_trips"] = activity["departures"] + activity["arrivals"]
    activity = activity.merge(stations, on="station_id", how="left")
    activity = activity.sort_values("station_trips", ascending=False)
else:
    activity = pd.DataFrame(columns=["station_id", "departures", "arrivals", "station_trips", "station_name", "lat", "lon"])

trips_all = int(daily_sel["trips"].sum())
station_trips = int(route_counts["trips"].sum())
number_stations = activity["station_id"].nunique()

with st.sidebar.expander("How to use this dashboard", expanded=False):
    st.write("Choose a date range and rider type. Switch tabs to compare when trips happen, where station activity occurs, and how stations connect. Hover or zoom on any chart to inspect values.")

intro, time_tab, map_tab, network_tab, method_tab = st.tabs([
    "📌 Overview", "📈 Time series", "🗺️ Station map", "🕸️ Station network", "📖 Findings & methods"
])

with intro:
    st.subheader("One mobility system, three perspectives")
    a, b, c, d = st.columns(4)
    a.metric("Trips in time series", f"{trips_all:,}")
    b.metric("Station-to-station trips", f"{station_trips:,}")
    c.metric("Stations in network data", f"{number_stations:,}")
    d.metric("Days selected", (end_date - start_date).days + 1)
    st.write("A trip contributes to the time view when its start/end timestamps and duration pass validation. "
             "A trip contributes to the map and network only when **both station IDs** are present. "
             "Those two totals are therefore different by design.")
    st.info("**Read the views:** Time series = WHEN people ride. Map = WHERE bike stations are. Network = WHICH station pairs have many recorded rides.")
    st.caption(f"Dataset period: {meta['date_start']} to {meta['date_end']}. Data month: {meta['month']}. The full source is historical, not live bike availability.")

with time_tab:
    st.subheader("When do people ride?")
    d = daily_sel.groupby("date", as_index=False)["trips"].sum().sort_values("date")
    if d.empty:
        st.warning("No trips for these filters.")
    else:
        fig = px.line(d, x="date", y="trips", markers=True, labels={"date": "Trip start date", "trips": "Number of trips"}, title="Daily bike-share trips")
        fig.update_traces(line_color="#2563eb")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"**Text alternative (daily line chart):** {len(d)} days displayed; minimum {int(d['trips'].min()):,} and maximum {int(d['trips'].max()):,} trips per day.")
        h = hourly_sel.groupby("hour", as_index=False)["trips"].sum()
        fig_h = px.bar(h, x="hour", y="trips", labels={"hour": "Local start hour (0–23)", "trips": "Number of trips"}, title="Trip start times, summed across selected dates")
        fig_h.update_traces(marker_color="#117c83")
        fig_h.update_xaxes(dtick=2)
        st.plotly_chart(fig_h, width="stretch")
        st.caption("**Text alternative (hourly bar chart):** Bars show ride starts for each local clock hour, summed over the selected days. The peak hour is named below.")
        peak_hour = int(h.loc[h["trips"].idxmax(), "hour"])
        peak_day = d.loc[d["trips"].idxmax(), "date"].strftime("%B %d, %Y")
        st.write(f"**What this view reveals:** The busiest start hour in the selection is **{peak_hour}:00**, and the busiest recorded day is **{peak_day}**. These are observed trip-start counts, not proof of commuting purpose.")
        st.caption("**What could mislead:** Daily totals cannot reveal why a trip occurred. Time fields have no explicit timezone; local daylight-saving transitions may affect clock-hour comparisons. A single month cannot show annual seasonality.")
        with st.expander("Accessible data table: daily trips"):
            st.dataframe(d, hide_index=True, width="stretch")

with map_tab:
    st.subheader("Where is recorded station activity concentrated?")
    max_stations = st.slider("Number of stations on map", min_value=10, max_value=100, value=40, step=10)
    m = activity.dropna(subset=["lat", "lon"]).head(max_stations).copy()
    if m.empty:
        st.warning("No stations with valid coordinates match this selection.")
    else:
        m["departure_share"] = (m["departures"] / m["station_trips"].replace(0, np.nan)).fillna(0)
        fig_map = px.scatter_map(
            m, lat="lat", lon="lon", size="station_trips", color="departure_share",
            hover_name="station_name", hover_data={"station_id": True, "departures": True, "arrivals": True,
                                                     "station_trips": True, "lat": False, "lon": False,
                                                     "departure_share": ":.1%"},
            color_continuous_scale="Viridis", size_max=24, zoom=11.5,
            center={"lat": 38.91, "lon": -77.07}, map_style="open-street-map",
            labels={"station_trips": "Arrivals + departures", "departure_share": "Departure share"},
            title=f"Top {len(m)} stations by combined departures + arrivals",
        )
        # Explicitly frame the Washington, DC region with an explicit zoom.
        fig_map.update_layout(
            map=dict(
                center={"lat": 38.91, "lon": -77.07},
                zoom=11.5,
                style="open-street-map",
            ),
            height=580,
            margin=dict(t=45, l=0, r=0, b=0),
        )
        st.plotly_chart(fig_map, width="stretch")
        st.caption(f"**Text alternative (map):** {len(m)} station locations are plotted on a basemap. Larger circles have more arrivals and departures, and color indicates departure share. A ranked station table appears below.")
        top = m.iloc[0]
        st.write(f"**What this view reveals:** **{top['station_name']}** has the most recorded station activity within the displayed selection ({int(top['station_trips']):,} arrivals plus departures). Marker **size** shows activity; color shows the share of activity that consists of departures.")
        st.caption("**What could mislead:** These are station locations, not actual routes. A ride touching a missing/dockless station is omitted here. A trip beginning and ending at the same station contributes to both departures and arrivals.")
        with st.expander("Accessible data table: station activity"):
            st.dataframe(m[["station_name", "station_id", "departures", "arrivals", "station_trips"]], hide_index=True, width="stretch")

with network_tab:
    st.subheader("Which stations connect through trips?")
    st.write("Stations are **nodes**; recorded trips between two stations are **weighted directed flows**. "
             "The visualization combines directions between each pair to reduce clutter. The underlying directed route counts are available below.")
    c1, c2, c3 = st.columns(3)
    with c1:
        limit_nodes = st.slider("Top stations included", min_value=10, max_value=50, value=25, step=5)
    with c2:
        threshold = st.slider("Minimum trips per station pair", min_value=1, max_value=100, value=10, step=1)
    with c3:
        layout_type = st.selectbox("Network layout", ["Force-directed", "Circular"])

    top_ids = set(activity.head(limit_nodes)["station_id"].dropna().astype(str))
    selected_edges = route_counts.loc[
        route_counts["start_id"].isin(top_ids) & route_counts["end_id"].isin(top_ids)
    ].copy()
    # For legibility: combine both directions in the diagram, but retain directional raw table.
    selected_edges = selected_edges.loc[selected_edges["start_id"] != selected_edges["end_id"]]
    selected_edges["a"] = selected_edges[["start_id", "end_id"]].min(axis=1)
    selected_edges["b"] = selected_edges[["start_id", "end_id"]].max(axis=1)
    undirected = (selected_edges.groupby(["a", "b"], as_index=False)["trips"].sum()
                  .query("trips >= @threshold").sort_values("trips", ascending=False).head(140))
    G = nx.Graph()
    for row in undirected.itertuples(index=False):
        G.add_edge(str(row.a), str(row.b), weight=int(row.trips))
    if G.number_of_edges() == 0:
        st.warning("No connections at this threshold. Try lowering the minimum-trip slider or choosing more stations.")
    else:
        positions = nx.spring_layout(G, seed=42, weight="weight") if layout_type == "Force-directed" else nx.circular_layout(G)
        community_sets = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        community_ids = {str(node): idx for idx, comm in enumerate(community_sets) for node in comm}
        degrees = dict(G.degree())
        station_names = dict(zip(stations["station_id"].astype(str), stations["station_name"].astype(str)))
        colors = px.colors.qualitative.Safe
        chart = go.Figure()
        max_w = max(float(np.log1p(d["weight"])) for _, _, d in G.edges(data=True))
        for u, v, attrs in G.edges(data=True):
            x0, y0 = positions[u]; x1, y1 = positions[v]
            weight = attrs["weight"]
            width = 0.8 + 3 * float(np.log1p(weight)) / max_w
            chart.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="#9aa5b1", width=width),
                hoverinfo="skip", showlegend=False,
            ))
        for comm_id, comm in enumerate(community_sets):
            nodes = sorted(comm)
            chart.add_trace(go.Scatter(
                x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes], mode="markers",
                marker=dict(size=[11 + 2.3*degrees[n] for n in nodes], color=colors[comm_id % len(colors)],
                            line=dict(width=1, color="#243040")),
                text=[station_names.get(n, n) for n in nodes],
                customdata=[[degrees[n], comm_id + 1] for n in nodes],
                hovertemplate="%{text}<br>Displayed degree: %{customdata[0]}<br>Community: %{customdata[1]}<extra></extra>",
                name=f"Community {comm_id + 1}",
            ))
        chart.update_layout(title=f"{layout_type} layout: {G.number_of_nodes()} stations, {G.number_of_edges()} connections",
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            height=590, hovermode="closest", showlegend=True,
                            legend=dict(orientation="h", y=-0.04),
                            margin=dict(t=60, l=10, r=10, b=60), plot_bgcolor="white")
        st.plotly_chart(chart, width="stretch")
        st.caption(f"**Text alternative (network):** {G.number_of_nodes()} station nodes and {G.number_of_edges()} weighted connections are drawn; larger nodes have more displayed neighbors, node colors indicate detected communities, and thicker lines indicate more rides. The associated node ranking is below.")
        biggest_degree = max(degrees, key=degrees.get)
        st.write(f"**What this view reveals:** **{station_names.get(biggest_degree, biggest_degree)}** has the most displayed station connections (degree {degrees[biggest_degree]}). "
                 "Node **size** represents degree, **color** represents greedy-modularity community, and line **thickness** represents trips between the two stations (both directions combined).")
        st.caption("**What could mislead:** This is only the top-station, thresholded subgraph; its degrees and communities are not ranks for the full network. Layout position is not geographic distance. Lines do not represent actual bicycle paths; two-way flows are merged for clarity.")
        ranks = pd.DataFrame({"station": [station_names.get(n, n) for n in G.nodes],
                              "displayed_degree": [degrees[n] for n in G.nodes],
                              "community": [community_ids[n] + 1 for n in G.nodes]})
        with st.expander("Accessible data table: displayed network nodes"):
            st.dataframe(ranks.sort_values("displayed_degree", ascending=False), hide_index=True, width="stretch")
    with st.expander("Directed station-pair trip counts (unmerged source data)"):
        st.dataframe(route_counts.sort_values("trips", ascending=False).head(100), hide_index=True, width="stretch")

with method_tab:
    st.subheader("Narrative and research method")
    st.markdown("""
    **Research question.** How do time, station geography, and ride connections together describe observed bike-share mobility?

    **Time series:** Date/time stamps reveal changes in observed trip starts. Read a peak as a usage pattern, not direct evidence of people's motivations.

    **Geographic map:** Latitude and longitude position stations in real-world space. Marker size indicates arrivals plus departures for trips where both stations are known. Map proximity alone does not imply that stations exchange many riders.

    **Network:** A station is a node; station-to-station rides form weighted edges. Diagram edges merge both directions to improve readability; the raw directed data remains available. Node size is the number of displayed connections. Node color is community membership calculated using greedy modularity. Node positions use a spring or circular layout, not actual coordinates.

    **Important limitations:**
    - The map and network exclude rides without both station IDs (including some dockless electric-bike rides). This is a selection bias, so the map totals are not the same as all trip totals.
    - Files record trip origins and destinations, not the streets people traveled; never interpret a connecting line as a physical cycling route.
    - The source timestamps have no explicit time zone; clock-hour comparisons around daylight-saving changes require caution.
    - The network highlights only selected high-activity stations and strong pairs. Removing lower-activity stations changes degree and community assignments.
    - An observed peak or station community is descriptive, not proof of commuting, causation, or rider identity.

    **Accessibility:** Every view has labeled variables, descriptive text, and an accessible table. Station communities are identified with both color and legend/hover text. The map uses a perceptually ordered color scale; information is not conveyed by color alone.

    **Reproducibility:** Run `python prepare_midterm_data.py` to download the cited monthly ZIP and produce the aggregate files in `data/bikeshare_midterm/`. Commit those small output files with the code. No live network call is needed when opening the deployed Streamlit app.
    """)
    st.markdown(f"**Data source:** [Official Capital Bikeshare trip history](https://capitalbikeshare.com/system-data). [Download month ZIP]({meta['source_url']}).")
    st.json({"source_month": meta["month"], "original_rows": meta["original_rows"],
             "valid_trips": meta["valid_trips"], "station_to_station_trips": meta["station_to_station_trips"],
             "excluded_from_station_network": meta["excluded_from_station_network"]})
    st.caption("All values update from the selected date range and rider category, except the source metadata describing the full downloaded month.")
