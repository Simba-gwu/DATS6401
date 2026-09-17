import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.title("Multivariate Visualization of Fuel Efficiency and Vehicle Performance")

st.caption(
    "Dataset source: Auto MPG dataset, originally from the UCI Machine Learning Repository; "
    "cleaned CSV accessed from seaborn-data GitHub."
)

st.write("""
This app explores the Auto MPG dataset using a correlation heatmap and PCA.
The goal is to understand relationships between car performance variables and
see whether cars from different origins show patterns in PCA space.
""")


# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

# Clean missing values
df = df.dropna().reset_index(drop=True)


# Numeric variables
numeric_cols = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year"
]


st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Number of rows and columns:", df.shape)


# Sidebar widgets
st.sidebar.header("Controls")

selected_features = st.sidebar.multiselect(
    "Choose numeric variables",
    numeric_cols,
    default=numeric_cols
)

color_by = st.sidebar.selectbox(
    "Color PCA points by",
    ["origin", "cylinders", "model_year"]
)


# Safety check
if len(selected_features) < 2:
    st.warning("Please select at least two numeric variables.")
    st.stop()


# Correlation heatmap
st.subheader("Correlation Heatmap")

corr = df[selected_features].corr()

fig_heatmap = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation Heatmap"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.write("""
The correlation heatmap shows relationships between the original variables.
Values close to +1 mean two variables increase together. Values close to -1 mean
one variable increases while the other decreases. Values near 0 mean there is
little linear relationship.
""")


# PCA
st.subheader("PCA Projection")

X = df[selected_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

pca_df[color_by] = df[color_by].astype(str)
pca_df["car_name"] = df["name"]
pca_df["mpg"] = df["mpg"]
pca_df["horsepower"] = df["horsepower"]
pca_df["weight"] = df["weight"]


pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100
total_var = pc1_var + pc2_var

st.write(f"PC1 explains {pc1_var:.2f}% of the variance.")
st.write(f"PC2 explains {pc2_var:.2f}% of the variance.")
st.write(f"Together, PC1 and PC2 explain {total_var:.2f}% of the variance.")


fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color=color_by,
    hover_data=["car_name", "mpg", "horsepower", "weight"],
    title=f"PCA Projection Colored by {color_by}"
)

fig_pca.update_layout(
    xaxis_title=f"PC1 ({pc1_var:.1f}% variance)",
    yaxis_title=f"PC2 ({pc2_var:.1f}% variance)"
)

st.plotly_chart(fig_pca, use_container_width=True)


# PCA loadings
st.subheader("PCA Loadings")

loadings = pd.DataFrame(
    pca.components_.T,
    index=selected_features,
    columns=["PC1", "PC2"]
)

st.dataframe(loadings)

st.write("""
The PCA loadings help explain what PC1 and PC2 represent.
Variables with large positive or negative loading values have stronger influence
on that principal component.
""")


# Interpretation section
st.subheader("Short Interpretation")

st.write("""
The correlation heatmap shows a clear group of size and power variables.
Cylinders, displacement, horsepower, and weight are strongly positively correlated,
meaning larger cars also tend to have larger engines, more horsepower, and more weight.
MPG is strongly negatively correlated with those variables, which means heavier and
more powerful cars tend to have lower fuel efficiency.

The PCA projection reduces the seven numeric car variables into two dimensions.
PC1 explains most of the variance and mainly represents a size/power versus
fuel-efficiency direction. Cars with higher PC1 values tend to be heavier, have
more cylinders, larger displacement, and higher horsepower, while cars with lower
PC1 values tend to have better MPG.

PC2 is mainly influenced by model year, based on the PCA loadings table. This means
PC2 helps capture differences between older and newer cars.

Coloring the PCA plot by origin shows that cars from the USA, Japan, and Europe
occupy somewhat different areas of the PCA space. USA cars appear more spread toward
the size/power side, while Japanese and European cars tend to appear more toward the
smaller and more fuel-efficient side. This analysis is exploratory and shows patterns,
but it does not prove causation.
""")