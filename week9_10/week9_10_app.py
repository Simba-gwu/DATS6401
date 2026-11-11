"""Week 9–10 — US Supercharger geography: layered map, two joins, shared filter.

Run from the DATS6401 repo root:
    python -m streamlit run week9_10/week9_10_app.py

Uses the original Week 10 point, population, and US Census state-boundary data.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent / "data"
SHAPEFILE = BASE / "us_states_2023" / "cb_2023_us_state_500k.shp"

# Plotly's USA state shapes cover these 50 states and Washington, DC.
US_CODES = set("""
AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS
MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC
""".split())

GROUP_COLORS = {
    "Q1 · lowest": "#dbeafe",
    "Q2 · low": "#a9d4f5",
    "Q3 · middle": "#6eafe0",
    "Q4 · high": "#357ebc",
    "Q5 · highest": "#164e8b",
}


@st.cache_data(show_spinner="Reading points, boundaries and population; checking joins…")
def prepare_geodata():
    # Read coordinates as numbers; FIPS codes as TEXT so 01 stays 01.
    chargers = pd.read_csv(BASE / "superchargers_us.csv")
    chargers["record_id"] = np.arange(len(chargers))
    chargers["lat"] = pd.to_numeric(chargers["lat"], errors="coerce")
    chargers["lon"] = pd.to_numeric(chargers["lon"], errors="coerce")
    valid_coords = (
        chargers["lat"].between(-90, 90) & chargers["lon"].between(-180, 180)
    )

    population = pd.read_csv(BASE / "state_stats.csv", dtype={"fips": "string"})
    population["fips"] = population["fips"].str.zfill(2)
    population["pop_total"] = pd.to_numeric(population["pop_total"], errors="coerce")

    states = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
    states["STATEFP"] = states["STATEFP"].astype(str).str.zfill(2)
    boundaries = states[["STATEFP", "STUSPS", "NAME", "geometry"]].copy()

    points = gpd.GeoDataFrame(
        chargers,
        geometry=gpd.points_from_xy(
            chargers["lon"].where(valid_coords),
            chargers["lat"].where(valid_coords),
        ),
        crs="EPSG:4326",
    )

    # SPATIAL JOIN: left join preserves locations that do NOT match a polygon.
    joined = gpd.sjoin(points, boundaries, how="left", predicate="within")
    duplicate_matches = int(joined.duplicated("record_id").sum())
    # In rare boundary overlaps, keep one matching polygon per original record.
    joined = (
        joined.sort_values(["record_id", "STATEFP"], na_position="last")
        .drop_duplicates("record_id")
        .sort_values("record_id")
        .reset_index(drop=True)
    )

    matched = int(joined["STATEFP"].notna().sum())
    unmatched = int(joined["STATEFP"].isna().sum())
    disagreement = joined.loc[
        joined["STUSPS"].notna()
        & (joined["state"].astype(str).str.upper() != joined["STUSPS"])
    ].copy()

    # COUNT: use the state ASSIGNED BY THE SPATIAL JOIN, not the original label.
    counts = (
        joined.dropna(subset=["STATEFP"])
        .groupby("STATEFP", as_index=False)
        .size()
        .rename(columns={"size": "supercharger_count"})
    )

    # Keep all boundary regions, including those with ZERO chargers.
    state_table = boundaries.drop(columns="geometry").merge(
        counts, on="STATEFP", how="left", validate="one_to_one"
    )
    state_table["supercharger_count"] = (
        state_table["supercharger_count"].fillna(0).astype(int)
    )

    # ATTRIBUTE JOIN: match state FIPS identifiers to external population table.
    state_table = state_table.merge(
        population[["fips", "state", "pop_total"]],
        left_on="STATEFP", right_on="fips", how="left",
        validate="one_to_one", indicator="population_join",
    )
    state_table["locations_per_100k"] = np.where(
        state_table["pop_total"].gt(0),
        state_table["supercharger_count"] / state_table["pop_total"] * 100000,
        np.nan,
    )

    # Final USA map scope: 50 states and DC with a valid population denominator.
    analysis = state_table.loc[
        state_table["STUSPS"].isin(US_CODES)
        & state_table["pop_total"].gt(0)
    ].copy()

    # National bins are calculated ONCE, so filtering doesn't redefine the legend.
    if analysis["locations_per_100k"].nunique() > 1:
        qcut, edges = pd.qcut(
            analysis["locations_per_100k"], q=5, duplicates="drop", retbins=True
        )
        n_classes = len(edges) - 1
        group_names = (
            list(GROUP_COLORS) if n_classes == 5
            else [f"Q{i+1} of {n_classes}" for i in range(n_classes)]
        )
        analysis["rate_group"] = qcut.cat.codes.map(
            {i: name for i, name in enumerate(group_names)}
        )
    else:
        n_classes, edges, group_names = 1, np.array([]), ["One class"]
        analysis["rate_group"] = "One class"

    quality = {
        "total": len(chargers),
        "matched": matched,
        "unmatched": unmatched,
        "invalid_coords": int((~valid_coords).sum()),
        "duplicates": duplicate_matches,
        "disagreements": len(disagreement),
        "population_matched": int((state_table["population_join"] == "both").sum()),
        "population_unmatched_us": int((state_table["STUSPS"].isin(US_CODES) & (state_table["population_join"] != "both")).sum()),
        "mapped_regions": len(analysis),
        "excluded_territories": sorted(state_table.loc[~state_table["STUSPS"].isin(US_CODES), "NAME"].dropna().tolist()),
        "excluded_points": int(joined["STUSPS"].notna().sum() - joined["STUSPS"].isin(US_CODES).sum()),
        "n_classes": n_classes,
        "edges": edges.tolist(),
    }

    # Geometry not needed after the join; makes caching and displaying simpler.
    joined_table = pd.DataFrame(joined.drop(columns="geometry"))
    return chargers, joined_table, state_table, analysis, quality


def layered_map(analysis, points, show_points=True):
    # Colors represent rate groups; a second trace shows the actual sites.
    group_names = list(dict.fromkeys(analysis["rate_group"].tolist()))
    palette = GROUP_COLORS if all(g in GROUP_COLORS for g in group_names) else {
        name: px.colors.sequential.Blues[min(i + 2, 8)]
        for i, name in enumerate(group_names)
    }
    fig = px.choropleth(
        analysis,
        locations="STUSPS", locationmode="USA-states", scope="usa",
        color="rate_group", color_discrete_map=palette,
        category_orders={"rate_group": list(palette)},
        hover_name="NAME",
        hover_data={
            "supercharger_count": True,
            "pop_total": ":,",
            "locations_per_100k": ":.3f",
            "rate_group": False,
            "STUSPS": False,
        },
        title="Supercharger locations per 100,000 residents",
    )
    if show_points and not points.empty:
        fig.add_trace(go.Scattergeo(
            lat=points["lat"], lon=points["lon"], mode="markers",
            marker=dict(size=3.5, color="#f59e0b", opacity=0.65),
            name="Actual Supercharger location",
            text=points["name"], hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_geos(scope="usa", projection_type="albers usa", visible=False)
        
    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=70, b=110),
        legend=dict(
            title="Rate quintile / layer",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        )
    )

    return fig


def comparison_chart(frame, metric, title):
    ascending = frame.sort_values(metric, ascending=True)
    fig = px.bar(
        ascending, x=metric, y="NAME", orientation="h",
        color_discrete_sequence=["#387fb8"],
        labels={"NAME": "State", metric: title},
        hover_data=["supercharger_count", "pop_total", "locations_per_100k"],
    )
    fig.update_layout(height=max(360, 28 * len(ascending) + 75),
                      margin=dict(l=0, r=5, t=15, b=25), showlegend=False)
    return fig


def main():
    st.set_page_config(page_title="US Supercharger Geography | Week 9–10",
                       page_icon="🔋", layout="wide")
    st.title("🔋 US Supercharger Geography Explorer")
    st.write("Where are Supercharger locations, and how do state rates differ after accounting for population?")

    required = [BASE / "superchargers_us.csv", BASE / "state_stats.csv", SHAPEFILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("Missing local files. Keep the original point CSV, state statistics CSV, and extracted Census boundary files in the week9_10/data directory.")
        st.code("\n".join(missing))
        st.stop()

    chargers, joined, all_states, analysis, quality = prepare_geodata()

    st.sidebar.header("Explore states")
    selected = st.sidebar.multiselect(
        "State filter (empty = all)",
        options=sorted(analysis["NAME"].tolist()),
        default=[],
        help="This SAME filter changes the map and both state comparison charts.",
    )
    show_points = st.sidebar.checkbox("Overlay location dots", value=True)

    visible = analysis.loc[analysis["NAME"].isin(selected)].copy() if selected else analysis.copy()
    visible_points = joined.loc[joined["STUSPS"].isin(visible["STUSPS"])].copy()
    st.caption("State selection is shared across Map and State comparison tabs. Empty selection displays all available states.")

    tab_map, tab_comparison, tab_joins = st.tabs(
        ["🗺️ Layered map", "📊 State comparison", "🔗 Joins & data quality"]
    )

    with tab_map:
        c1, c2, c3 = st.columns(3)
        c1.metric("Locations in selected states", f"{len(visible_points):,}")
        c2.metric("States displayed", f"{len(visible):,}")
        c3.metric("Unmatched points (entire dataset)", f"{quality['unmatched']:,}")
        if not visible.empty:
            st.plotly_chart(layered_map(visible, visible_points, show_points), width="stretch")
        st.caption(
            f"**Layers:** state polygons = recorded Supercharger locations per 100,000 residents "
            f"(five national rate quantiles; {quality['n_classes']} classes actually formed); "
            f"orange dots = individual Supercharger locations" + (" (hidden by user)" if not show_points else "") + ". "
            "**Projection:** Albers USA (Plotly `geo`, no tiled basemap); the displayed state outlines are Plotly built-ins. "
            "**Classification:** quantiles calculated over the full mapped set of states, "
            "not recalculated after filtering. "
            f"**Excluded from plotted dots:** {quality['unmatched']} spatially unmatched point(s) "
            f"and {quality['excluded_points']} point(s) assigned outside the 50-state/DC scope. "
            "**Units:** locations, not charging stalls."
        )
        st.info("The spatial join uses Census 2023 state polygons in EPSG:4326. The display uses Plotly's built-in USA state outlines in Albers USA projection.")

    with tab_comparison:
        st.subheader("Compare raw counts with population-adjusted rates")
        ranked = visible.sort_values("locations_per_100k", ascending=False)
        if not selected:
            ranked = ranked.head(15)
            st.caption("Showing the 15 highest-rate states. Choose specific states in the sidebar to compare any state.")
        else:
            st.caption("Showing all selected states. The sidebar filter also updates the map.")
        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("**Locations per 100,000 residents**")
            st.plotly_chart(comparison_chart(ranked, "locations_per_100k", "Locations per 100k"), width="stretch")
        with chart2:
            st.markdown("**Raw location count**")
            st.plotly_chart(comparison_chart(ranked, "supercharger_count", "Number of locations"), width="stretch")
        st.markdown("**State figures and denominators**")
        st.dataframe(
            visible[["NAME", "supercharger_count", "pop_total", "locations_per_100k"]]
            .rename(columns={"NAME": "State", "supercharger_count": "Locations", "pop_total": "Population", "locations_per_100k": "Locations per 100k"})
            .sort_values("Locations per 100k", ascending=False),
            hide_index=True, width="stretch",
        )
        st.caption("Rate = (location count ÷ state population) × 100,000. This is a per-resident rate, not a measure of charging capacity, geographic accessibility, or expected demand.")

    with tab_joins:
        st.subheader("1. Spatial join: points → polygons")
        a, b, c = st.columns(3)
        a.metric("Input point records", f"{quality['total']:,}")
        b.metric("Matched to a state", f"{quality['matched']:,}")
        c.metric("No polygon match", f"{quality['unmatched']:,}")
        st.write(
            "Points were built from longitude (x) and latitude (y) in WGS 84 (EPSG:4326). "
            "Using `geopandas.sjoin(..., how='left', predicate='within')`, each point was "
            "assigned to a Census 2023 state boundary, and unmatched rows were retained."
        )
        st.caption(
            f"Invalid/out-of-range coordinates: {quality['invalid_coords']}. "
            f"Additional matches from polygon overlaps: {quality['duplicates']} "
            "(one assignment kept per point if overlap occurred). "
            f"Recorded-state vs spatial-state disagreements: {quality['disagreements']}."
        )
        st.dataframe(
            joined[["name", "state", "lat", "lon", "NAME", "STUSPS"]]
            .head(12).rename(columns={"name": "Site", "state": "Original state", "NAME": "Spatial state", "STUSPS": "Spatial abbreviation"}),
            width="stretch", hide_index=True,
        )
        if quality["unmatched"] == 0:
            st.success("No unmatched Supercharger points in the supplied dataset under these boundaries and the 'within' test.")
        else:
            st.warning("Unmatched points can be caused by missing/invalid coordinates, a point lying on a boundary, or the location being outside the covered polygons. Inspect these records before deciding the reason.")
            st.dataframe(joined.loc[joined["STUSPS"].isna(), ["name", "state", "lat", "lon"]], width="stretch", hide_index=True)

        st.subheader("2. Attribute join: state counts → population")
        st.write("We counted the spatially assigned sites for each state, then joined those counts to the external `state_stats.csv` using two-character **state FIPS** codes (`STATEFP` ↔ `fips`). States with no sites are retained with count 0.")
        e1, e2 = st.columns(2)
        e1.metric("Population matches (all boundary regions)", quality["population_matched"])
        e2.metric("Unmatched population joins (50 states + DC)", quality["population_unmatched_us"])
        st.dataframe(
            all_states[["NAME", "STATEFP", "supercharger_count", "pop_total", "locations_per_100k", "population_join"]]
            .rename(columns={"NAME": "Boundary region", "STATEFP": "FIPS", "supercharger_count": "Sites", "pop_total": "Population", "locations_per_100k": "Per 100k", "population_join": "Population join"})
            .sort_values("Boundary region"),
            hide_index=True, width="stretch",
        )
        st.caption(
            "The Census boundary file also includes US territories. The primary Plotly USA-state "
            "map excludes those territories; their presence without a population match "
            "is not a failed match for one of the 50 states/DC. "
            "Territories excluded from the map: "
            + (", ".join(quality["excluded_territories"]) if quality["excluded_territories"] else "none") + "."
        )
        st.markdown("**Sources and limitations**")
        st.markdown(
            "- Charger points and population: course Week 10 files. "
            "[Supercharger CSV](https://gaimsds.github.io/dataviz/dats6401/weekly/week10/data/superchargers_us.csv).\n"
            "- Boundaries: US Census 2023 cartographic state boundary shapefile (500k scale).\n"
            "- Rates normalize by total resident population; they do not capture EV ownership, site capacity, access distance, or charging demand.\n"
            "- Data represent the provided course snapshot, not a live charger inventory."
        )


if __name__ == "__main__":
    main()
