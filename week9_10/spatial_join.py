
import pandas as pd
import geopandas as gpd

# 1. Read the Supercharger dataset
chargers = pd.read_csv("data/superchargers_us.csv")

# 2. Read the US state boundaries
states = gpd.read_file(
    "data/us_states_2023/cb_2023_us_state_500k.shp"
)

# 3. Use the same coordinate system for both datasets
states = states.to_crs("EPSG:4326")

# 4. Convert latitude and longitude into geographic points
points = gpd.GeoDataFrame(
    chargers,
    geometry=gpd.points_from_xy(
        chargers["lon"],
        chargers["lat"]
    ),
    crs="EPSG:4326"
)

# 5. Spatial join: find which state contains each point
joined = gpd.sjoin(
    points,
    states[["NAME", "STUSPS", "geometry"]],
    how="left",
    predicate="within"
)

# 6. Count successful and unsuccessful matches
matched = joined["STUSPS"].notna().sum()
unmatched = joined["STUSPS"].isna().sum()

print("Total Supercharger locations:", len(chargers))
print("Matched to a state:", matched)
print("Unmatched locations:", unmatched)

# 7. Show some results
print("\nFirst five spatial join results:")
print(
    joined[
        ["name", "state", "STUSPS", "NAME"]
    ].head()
)

# 8. Compare recorded states with spatially assigned states
disagreements = joined[
    joined["STUSPS"].notna()
    & (joined["state"] != joined["STUSPS"])
]

print("\nState label disagreements:", len(disagreements))


# STEP 9: Count Supercharger locations in each state
state_counts = (
    joined.groupby("NAME")
    .size()
    .reset_index(name="supercharger_count")
)

# STEP 10: Read state population data
population = pd.read_csv(
    "data/state_stats.csv",
    dtype={"fips": str}
)

# STEP 11: Attribute join using state names
state_summary = state_counts.merge(
    population,
    left_on="NAME",
    right_on="state",
    how="left",
    validate="one_to_one",
    indicator=True
)

# STEP 12: Check states without population matches
unmatched_population = state_summary[
    state_summary["_merge"] != "both"
]

print("\nStates without population match:", len(unmatched_population))

if not unmatched_population.empty:
    print(unmatched_population["NAME"].tolist())

# STEP 13: Calculate locations per 100,000 residents
state_summary["locations_per_100k"] = (
    state_summary["supercharger_count"]
    / state_summary["pop_total"]
    * 100000
)

# STEP 14: Display top 10 states
print("\nTop 10 states by Supercharger locations per 100,000 people:")

print(
    state_summary[
        ["NAME", "supercharger_count", "pop_total", "locations_per_100k"]
    ]
    .sort_values("locations_per_100k", ascending=False)
    .head(10)
    .to_string(index=False)
)

# STEP 15: Save the results
state_summary.drop(columns=["_merge"]).to_csv(
    "data/state_summary.csv",
    index=False
)


# STEP 16: Find states missing from the charger-count table

states_without_chargers = population[
    ~population["state"].isin(state_counts["NAME"])
]

print("\nStates with no recorded Supercharger locations:")
print(
    states_without_chargers[
        ["fips", "state", "pop_total"]
    ].to_string(index=False)
)

print(
    "\nNumber of states without chargers:",
    len(states_without_chargers)
)
