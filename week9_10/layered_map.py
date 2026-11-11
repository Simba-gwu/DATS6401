
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# Load Supercharger locations
chargers = pd.read_csv("data/superchargers_us.csv")

# Load our state-level results
summary = pd.read_csv(
    "data/state_summary.csv",
    dtype={"fips": str}
)

# Load state names and abbreviations
states = gpd.read_file(
    "data/us_states_2023/cb_2023_us_state_500k.shp"
)

# Connect results to state abbreviations
map_data = states[["NAME", "STUSPS"]].merge(
    summary,
    on="NAME",
    how="inner"
)

# Divide states into 5 groups based on their rates
labels = [
    "Lowest 20%",
    "Low",
    "Middle",
    "High",
    "Highest 20%"
]

map_data["rate_group"] = pd.qcut(
    map_data["locations_per_100k"],
    q=5,
    labels=labels
)

# Layer 1: State colors
fig = px.choropleth(
    map_data,
    locations="STUSPS",
    locationmode="USA-states",
    color="rate_group",
    color_discrete_sequence=px.colors.sequential.Blues[1:6],
    category_orders={"rate_group": labels},
    hover_name="NAME",
    hover_data={
        "supercharger_count": True,
        "locations_per_100k": ":.2f",
        "rate_group": False,
        "STUSPS": False
    },
    scope="usa",
    title="US Supercharger Locations per 100,000 Residents"
)

# Layer 2: Individual Supercharger locations
fig.add_trace(
    go.Scattergeo(
        lat=chargers["lat"],
        lon=chargers["lon"],
        mode="markers",
        marker=dict(
            size=3,
            color="orange",
            opacity=0.55
        ),
        name="Supercharger location",
        text=chargers["name"],
        hovertemplate="%{text}<extra></extra>"
    )
)

# Use the Albers USA map projection
fig.update_geos(
    scope="usa",
    projection_type="albers usa"
)

fig.update_layout(
    height=700,
    legend_title_text="Rate group",
    margin=dict(l=0, r=0, t=65, b=30)
)

# Display the map
fig.show()
