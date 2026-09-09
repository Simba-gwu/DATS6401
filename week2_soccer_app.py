import streamlit as st
import pandas as pd
import altair as alt


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Premier League Player Performance Analysis",
    layout="centered",
)

st.title("⚽ Premier League Player Performance Analysis — 2024/25")

st.markdown("""
This assignment uses match-level player statistics from the **2024/25 English
Premier League**. The dataset covers matches from **August 16, 2024 through
December 5, 2024**.

Each original row represents a player's performance in a match. The match-level
records are aggregated to create one observation per player containing total
minutes, goals, assists, and goal contributions.
""")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("data/premier_league_24_25.csv")

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------
st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Match-Player Records", len(df))
col2.metric("Players", df["Player"].nunique())
col3.metric("Teams", df["Team"].nunique())
col4.metric("Variables", df.shape[1])

st.write(
    f"The dataset contains **{len(df):,} match-player records**, "
    f"covering **{df['Player'].nunique()} players** and "
    f"**{df['Team'].nunique()} Premier League teams**."
)

with st.expander("View a preview of the raw dataset"):
    st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------------------------------
# Position cleaning
# ---------------------------------------------------------

# Take the first listed position as the player's primary
# position for each match.
df["Primary Position"] = (
    df["Position"]
    .str.split(",")
    .str[0]
    .str.strip()
)

# Determine the most frequently played primary position
# for each player.
player_positions = (
    df.groupby("Player")["Primary Position"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)


def broad_position(position):
    if position == "GK":
        return "Goalkeeper"

    elif position in ["CB", "LB", "RB", "WB"]:
        return "Defender"

    elif position in ["DM", "CM", "AM", "LM", "RM"]:
        return "Midfielder"

    elif position in ["FW", "LW", "RW"]:
        return "Forward"

    else:
        return "Other"


player_positions["Position Group"] = (
    player_positions["Primary Position"]
    .apply(broad_position)
)


# ---------------------------------------------------------
# Aggregate match records to player level
# ---------------------------------------------------------
player_stats = (
    df.groupby("Player", as_index=False)
    .agg(
        Team=("Team", lambda x: x.mode().iloc[0]),
        Minutes=("Minutes", "sum"),
        Goals=("Goals", "sum"),
        Assists=("Assists", "sum")
    )
)

player_stats = player_stats.merge(
    player_positions,
    on="Player",
    how="left"
)

# Create attacking-output variable
player_stats["Goal Contributions"] = (
    player_stats["Goals"] + player_stats["Assists"]
)


# ---------------------------------------------------------
# Keep outfield players
# ---------------------------------------------------------
outfield_players = player_stats[
    player_stats["Position Group"].isin(
        ["Defender", "Midfielder", "Forward"]
    )
].copy()


# ---------------------------------------------------------
# Data preparation explanation
# ---------------------------------------------------------
st.subheader("Data Preparation")

st.markdown("""
Because the raw dataset contains multiple match records for each player,
the data was grouped by player before visualization.

A player's most frequently occurring primary position was used to assign
them to one of three broad outfield roles:

- **Defender:** CB, LB, RB, WB
- **Midfielder:** DM, CM, AM, LM, RM
- **Forward:** FW, LW, RW

**Goal Contributions** were calculated as:

**Goals + Assists**

Goalkeepers were excluded because the research question focuses on attacking
output among outfield players.
""")


# ---------------------------------------------------------
# Research question
# ---------------------------------------------------------
st.header("Research Question")

st.markdown("""
### Among Premier League outfield players, how is playing time related to
goal contributions, and does this relationship differ between defenders,
midfielders, and forwards?
""")

st.write(
    f"The analysis contains **{len(outfield_players)} outfield players**: "
    f"{(outfield_players['Position Group'] == 'Defender').sum()} defenders, "
    f"{(outfield_players['Position Group'] == 'Midfielder').sum()} midfielders, "
    f"and {(outfield_players['Position Group'] == 'Forward').sum()} forwards."
)


# ---------------------------------------------------------
# Chart 1 — Color
# ---------------------------------------------------------
st.header("Chart 1: Position Encoded by Color")

chart1 = (
    alt.Chart(outfield_players)
    .mark_circle(size=70, opacity=0.75)
    .encode(
        x=alt.X(
            "Minutes:Q",
            title="Total Minutes Played",
            scale=alt.Scale(zero=True)
        ),

        y=alt.Y(
            "Goal Contributions:Q",
            title="Goals + Assists",
            scale=alt.Scale(zero=True)
        ),

        color=alt.Color(
            "Position Group:N",
            title="Player Position",
            sort=["Defender", "Midfielder", "Forward"],
            scale=alt.Scale(scheme="set2")
        ),

        tooltip=[
            alt.Tooltip("Player:N", title="Player"),
            alt.Tooltip("Team:N", title="Team"),
            alt.Tooltip("Position Group:N", title="Position"),
            alt.Tooltip("Minutes:Q", title="Minutes"),
            alt.Tooltip("Goals:Q", title="Goals"),
            alt.Tooltip("Assists:Q", title="Assists"),
            alt.Tooltip(
                "Goal Contributions:Q",
                title="Goal Contributions"
            )
        ]
    )
    .properties(
        title="Playing Time vs. Goal Contributions"
    )
    .interactive()
)

st.altair_chart(chart1, use_container_width=True)

st.markdown("""
**Encoding rationale:**  
Total minutes played and goal contributions are encoded using **x- and
y-position** because position is one of the most perceptually accurate
channels for comparing quantitative values.

Player position is a **nominal categorical variable**, so color hue is used
to distinguish defenders, midfielders, and forwards.

**Perceptual / Gestalt principle:**  
The Gestalt principle of **similarity** causes points with the same color to
be perceived as belonging to the same group. This makes the three positional
groups easy to identify while preserving the overall relationship between
playing time and attacking output.
""")


# ---------------------------------------------------------
# Chart 2 — Shape
# ---------------------------------------------------------
st.header("Chart 2: Position Encoded by Shape")

chart2 = (
    alt.Chart(outfield_players)
    .mark_point(size=90)
    .encode(
        x=alt.X(
            "Minutes:Q",
            title="Total Minutes Played",
            scale=alt.Scale(zero=True)
        ),

        y=alt.Y(
            "Goal Contributions:Q",
            title="Goals + Assists",
            scale=alt.Scale(zero=True)
        ),

        shape=alt.Shape(
            "Position Group:N",
            title="Player Position",
            sort=["Defender", "Midfielder", "Forward"]
        ),

        tooltip=[
            alt.Tooltip("Player:N", title="Player"),
            alt.Tooltip("Team:N", title="Team"),
            alt.Tooltip("Position Group:N", title="Position"),
            alt.Tooltip("Minutes:Q", title="Minutes"),
            alt.Tooltip("Goals:Q", title="Goals"),
            alt.Tooltip("Assists:Q", title="Assists"),
            alt.Tooltip(
                "Goal Contributions:Q",
                title="Goal Contributions"
            )
        ]
    )
    .properties(
        title="Playing Time vs. Goal Contributions"
    )
    .interactive()
)

st.altair_chart(chart2, use_container_width=True)

st.markdown("""
**Encoding rationale:**  
Total minutes and goal contributions remain encoded using x- and y-position,
while player position is represented using **shape instead of color**.

**Perceptual / Gestalt principle:**  
Shape similarity allows players with the same positional group to be perceived
as belonging together. Shape also provides an alternative to color and remains
useful when the visualization is viewed in grayscale.

However, identifying several shapes generally requires more visual effort than
distinguishing colors.
""")


# ---------------------------------------------------------
# Chart 3 — Small Multiples
# ---------------------------------------------------------
st.header("Chart 3: Position Encoded with Small Multiples")

chart3 = (
    alt.Chart(outfield_players)
    .mark_circle(size=70, opacity=0.75)
    .encode(
        x=alt.X(
            "Minutes:Q",
            title="Total Minutes Played",
            scale=alt.Scale(zero=True)
        ),

        y=alt.Y(
            "Goal Contributions:Q",
            title="Goals + Assists",
            scale=alt.Scale(zero=True)
        ),

        tooltip=[
            alt.Tooltip("Player:N", title="Player"),
            alt.Tooltip("Team:N", title="Team"),
            alt.Tooltip("Position Group:N", title="Position"),
            alt.Tooltip("Minutes:Q", title="Minutes"),
            alt.Tooltip("Goals:Q", title="Goals"),
            alt.Tooltip("Assists:Q", title="Assists"),
            alt.Tooltip(
                "Goal Contributions:Q",
                title="Goal Contributions"
            )
        ]
    )
    .properties(
        width=250,
        height=300
    )
    .facet(
        column=alt.Column(
            "Position Group:N",
            title=None,
            sort=["Defender", "Midfielder", "Forward"]
        )
    )
    .resolve_scale(
        x="shared",
        y="shared"
    )
)

st.altair_chart(chart3, use_container_width=True)

st.markdown("""
**Encoding rationale:**  
Total minutes and goal contributions remain encoded using x- and y-position.
Instead of representing player position through color or shape, the three
positions are separated into individual panels.

**Perceptual / Gestalt principle:**  
Small multiples use **proximity and grouping** by placing players with the
same positional role together.

Separating the groups reduces overlap and clutter, while the shared x- and
y-scales ensure that defenders, midfielders, and forwards can be compared
fairly.
""")


# ---------------------------------------------------------
# Comparison and conclusion
# ---------------------------------------------------------
st.header("Comparison and Conclusion")

st.markdown("""
All three visualizations examine the same underlying question while changing
the visual encoding used for player position.

### Chart 1 — Color
Color allows the three positional groups to be identified quickly while
keeping every player in one visualization. However, overlapping observations
can make dense areas difficult to interpret.

### Chart 2 — Shape
Shape successfully separates the positional groups without relying on color,
which can improve accessibility in grayscale environments. However, identifying
the shapes requires more visual effort than identifying the colors in Chart 1.

### Chart 3 — Small Multiples
Small multiples separate defenders, midfielders, and forwards into individual
panels. This reduces overlap and makes the distributions of the three groups
easier to compare. Shared axes preserve a consistent quantitative comparison.

### Conclusion
For this research question, the **small-multiple visualization is the most
effective design** because it reduces visual clutter while retaining accurate
x- and y-position encodings.

Across the three visualizations, greater playing time is generally
**associated with** more goal contributions. The relationship also differs by
position: forwards tend to produce the highest attacking output, midfielders
show moderate attacking output, and defenders are concentrated at lower
goal-contribution values.

Because goals and assists are cumulative statistics, players with more minutes
also have more opportunities to accumulate goal contributions. Therefore,
these visualizations describe an association rather than establishing a causal
effect of playing time.
""")