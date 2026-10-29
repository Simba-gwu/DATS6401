"""DATS 6401 Week 8: interactive EuroSAT RGB image explorer.

Run from anywhere with: python -m streamlit run week8/week8_app.py
First run prepare_week8_sample.py to generate a small image sample and counts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sample"
COUNTS_FILE = BASE / "data" / "class_counts.csv"
CHANNEL_NAMES = ["Red", "Green", "Blue"]
CHANNEL_COLORS = ["#d75a5a", "#329c61", "#4779d4"]

st.set_page_config(
    page_title="EuroSAT Image Explorer | Week 8",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ EuroSAT RGB Image Explorer")
st.write(
    "How do satellite images from different land-cover categories vary "
    "in appearance, color intensity, and visible edges?"
)

if not COUNTS_FILE.is_file() or not DATA.is_dir():
    st.error("Sample dataset not found. Run `python prepare_week8_sample.py` from week8 first.")
    st.stop()

counts = pd.read_csv(COUNTS_FILE)
classes = counts["category"].tolist()
paths_by_class = {
    category: sorted((DATA / category).glob("*.jpg")) for category in classes
}


@st.cache_data(show_spinner=False)
def image_array(path_string: str) -> np.ndarray:
    with Image.open(path_string) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def sampled_pixels(category: str, n: int) -> np.ndarray:
    paths = sorted((DATA / category).glob("*.jpg"))[:n]
    if not paths:
        return np.empty((0, 3), dtype=np.uint8)
    return np.concatenate([image_array(str(p)).reshape(-1, 3) for p in paths], axis=0)


def plot_rgb_histogram(image: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for channel, label in enumerate(CHANNEL_NAMES):
        freq = np.bincount(image[:, :, channel].ravel(), minlength=256)
        fig.add_trace(go.Bar(
            x=np.arange(256), y=freq, name=label,
            marker_color=CHANNEL_COLORS[channel], opacity=0.60,
        ))
    fig.update_layout(
        barmode="overlay", height=360,
        xaxis_title="Pixel intensity (0–255)",
        yaxis_title="Number of pixels",
        legend_title="RGB channel", margin=dict(t=20),
    )
    fig.update_xaxes(range=[0, 255])
    return fig


def plot_class_comparison(category_a: str, category_b: str, n: int):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{category_a} ({n} images)", f"{category_b} ({n} images)"],
        shared_yaxes=True,
    )
    means = []
    for column, category in enumerate([category_a, category_b], start=1):
        pixels = sampled_pixels(category, n)
        if not len(pixels):
            return None, None
        means.append([category] + [round(float(x), 1) for x in pixels.mean(axis=0)])
        for channel, name in enumerate(CHANNEL_NAMES):
            histogram = np.bincount(pixels[:, channel], minlength=256)
            # Percent of pixels per intensity value: comparable across classes.
            percent = 100 * histogram / len(pixels)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(256), y=percent,
                    mode="lines", name=name,
                    line=dict(color=CHANNEL_COLORS[channel], width=2),
                    showlegend=(column == 1),
                    hovertemplate="Intensity %{x}<br>Pixels %{y:.2f}%<extra>" + name + "</extra>",
                ),
                row=1, col=column,
            )
    fig.update_xaxes(title_text="Pixel intensity (0–255)", range=[0, 255])
    fig.update_yaxes(title_text="Pixels at this intensity (%)", row=1, col=1)
    fig.update_layout(height=440, legend_title="RGB channel", margin=dict(t=55))
    table = pd.DataFrame(means, columns=["Category", "Mean red", "Mean green", "Mean blue"])
    return fig, table


def sobel_edge_map(image: np.ndarray) -> np.ndarray:
    """Fixed 3x3 Sobel convolution: image edges, NOT AI-model saliency."""
    gray = (image.astype(float) * np.array([0.299, 0.587, 0.114])).sum(axis=2)
    p = np.pad(gray, 1, mode="edge")
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        - 2 * p[1:-1, :-2] + 2 * p[1:-1, 2:]
        - p[2:, :-2] + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
        + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
    )
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = 255 * magnitude / magnitude.max()
    return magnitude.astype(np.uint8)


overview_tab, montage_tab, rgb_tab, feature_tab = st.tabs([
    "📊 Dataset overview", "🖼️ Image montage", "🌈 RGB analysis", "🔎 Feature map & findings"
])

with overview_tab:
    total_images = int(counts["total_images"].sum())
    sample_images = int(counts["sampled_images"].sum())
    a, b, c = st.columns(3)
    a.metric("Original images", f"{total_images:,}")
    b.metric("Land-cover classes", len(classes))
    c.metric("Images in this app", sample_images)

    st.subheader("Are the classes balanced?")
    fig_counts = px.bar(
        counts.sort_values("total_images", ascending=False),
        x="category", y="total_images",
        labels={"category": "Land-cover category", "total_images": "Images in original dataset"},
        text="total_images",
        title="EuroSAT class balance (all original images)",
    )
    fig_counts.update_traces(textposition="outside")
    fig_counts.update_layout(height=480, margin=dict(t=60, b=100))
    fig_counts.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_counts, use_container_width=True)
    st.caption(
        "Counts are from all original images. The image browser uses a reproducible "
        "30-image-per-class sample so it runs online without uploading 27,000 files."
    )
    st.dataframe(counts, use_container_width=True, hide_index=True)

with montage_tab:
    st.subheader("Compare land-cover examples")
    group_choice = st.selectbox("Which categories should be shown?", ["All categories"] + classes)
    per_class = st.slider("Images per category", min_value=1, max_value=6, value=3)
    shown_classes = classes if group_choice == "All categories" else [group_choice]
    for category in shown_classes:
        paths = paths_by_class[category][:per_class]
        st.markdown(f"**{category}**")
        columns = st.columns(len(paths)) if paths else []
        for col, path in zip(columns, paths):
            col.image(image_array(str(path)), caption=path.name, use_container_width=True)
    st.info(
        "What the montage reveals: categories can differ in patterns and texture, "
        "such as regular roads in Residential areas and more uniform surfaces in SeaLake. "
        "Three examples per class are illustrative, not proof about every image."
    )

with rgb_tab:
    st.subheader("Explore the RGB values in one satellite image")
    category = st.selectbox("Select image category", classes, index=classes.index("Forest") if "Forest" in classes else 0)
    paths = paths_by_class[category]
    image_number = st.slider("Select image number", 1, len(paths), 1)
    chosen_image = image_array(str(paths[image_number - 1]))
    left, right = st.columns([1, 2])
    with left:
        st.image(chosen_image, caption=f"{category}: {paths[image_number - 1].name}", use_container_width=True)
        st.caption(f"Image shape: {chosen_image.shape} (height × width × RGB channels)")
    with right:
        st.plotly_chart(plot_rgb_histogram(chosen_image), use_container_width=True)
    st.caption(
        "Each channel is an intensity from 0 (darkest) to 255 (brightest). "
        "The entire 0–255 x-axis is displayed, even if no pixels are that bright."
    )

    st.divider()
    st.subheader("Compare the colors of two classes")
    x, y, z = st.columns([2, 2, 1])
    with x:
        class_a = st.selectbox("First category", classes,
                               index=classes.index("Forest") if "Forest" in classes else 0)
    with y:
        class_b = st.selectbox("Second category", classes,
                               index=classes.index("SeaLake") if "SeaLake" in classes else min(1, len(classes) - 1))
    with z:
        n_compare = st.slider("Images per class", 5, 30, 20, 5)

    comparison, mean_table = plot_class_comparison(class_a, class_b, n_compare)
    if comparison is not None:
        st.plotly_chart(comparison, use_container_width=True)
        st.write("**Average RGB intensity in the selected samples:**")
        st.dataframe(mean_table, hide_index=True, use_container_width=True)
    st.caption(
        "Each curve shows the percentage of selected pixels at each intensity. "
        "These distributions describe color but not spatial texture. Choose different "
        "classes to explore; sampled results are not a trained classifier."
    )

with feature_tab:
    st.subheader("Optional feature map: image edges")
    edge_category = st.selectbox("Choose category for the edge map", classes,
                                 index=classes.index("Residential") if "Residential" in classes else 0)
    edge_paths = paths_by_class[edge_category]
    edge_number = st.slider("Choose example", 1, len(edge_paths), 1, key="edge_number")
    edge_image = image_array(str(edge_paths[edge_number - 1]))
    edges = sobel_edge_map(edge_image)
    c1, c2 = st.columns(2)
    c1.image(edge_image, caption="Original RGB image", use_container_width=True)
    c2.image(edges, caption="Sobel edge-strength feature map", clamp=True, use_container_width=True)
    st.write(
        "**Interpretation:** Brighter regions in the feature map indicate stronger "
        "local changes in brightness. Roads, building boundaries, and field edges "
        "may appear strongly. Smooth water surfaces may have fewer strong edges. "
        "Try a few images and compare what changes."
    )
    st.warning(
        "This is a fixed Sobel convolution feature map, not saliency from a trained AI model. "
        "It shows detected boundaries, not what a classifier paid attention to."
    )

    st.subheader("Key findings and limitations")
    st.markdown(
        """
- **Montage:** The ten categories differ visually in geometry and texture. Some can share similar dark colors.
- **RGB:** Comparing Forest and SeaLake tests whether similar-looking images have different color distributions; results describe sampled pixels, not all images.
- **Structure:** RGB histograms ignore where pixels appear, so similar color distributions can hide very different spatial patterns.
- **Feature map:** Edge filters highlight local intensity changes but do not explain AI decisions.
- **Limitations:** Only RGB JPEG bands are shown, a fixed sample is used in the hosted app, and no model is trained or evaluated here.
"""
    )
    st.markdown(
        "**Source:** EuroSAT RGB (Sentinel-2), Helber et al. "
        "[Dataset and citation](https://zenodo.org/records/7711810). "
        "Original dataset: 27,000 labeled images in 10 land-cover classes."
    )
