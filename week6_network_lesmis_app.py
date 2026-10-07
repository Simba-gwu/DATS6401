import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities


# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Les Misérables Network Explorer",
    layout="wide"
)

st.title("Les Misérables Character Network Explorer")

st.write(
    """
    This Streamlit app explores a character co-occurrence network from *Les Misérables*.
    Each node represents a character, and each edge represents characters appearing together.
    Larger nodes represent more connected characters, node colors represent detected communities,
    and thicker edges represent stronger co-occurrence relationships.
    """
)


# --------------------------------------------------
# Cached data and network metrics
# --------------------------------------------------
@st.cache_data
def load_les_mis_network():
    """Load the built-in Les Misérables graph and calculate network metrics."""
    G = nx.les_miserables_graph()

    # Community detection using greedy modularity.
    communities = list(greedy_modularity_communities(G, weight="weight"))
    community_map = {}
    for community_id, community_nodes in enumerate(communities, start=1):
        for node in community_nodes:
            community_map[node] = community_id

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")

    node_rows = []
    for node in G.nodes():
        node_rows.append({
            "node": node,
            "degree": degree[node],
            "weighted_degree": weighted_degree[node],
            "betweenness": betweenness[node],
            "closeness": closeness[node],
            "pagerank": pagerank[node],
            "community": community_map[node],
        })

    nodes_df = pd.DataFrame(node_rows).sort_values(
        ["degree", "weighted_degree"], ascending=False
    )

    edge_rows = []
    for source, target, data in G.edges(data=True):
        edge_rows.append({
            "source": source,
            "target": target,
            "weight": data.get("weight", 1),
            "source_community": community_map[source],
            "target_community": community_map[target],
        })

    edges_df = pd.DataFrame(edge_rows).sort_values("weight", ascending=False)

    return G, nodes_df, edges_df


G, nodes_df, edges_df = load_les_mis_network()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Network Controls")

layout_choice = st.sidebar.selectbox(
    "Choose layout",
    ["Force-directed", "Circular"],
    index=0
)

size_metric = st.sidebar.selectbox(
    "Node size represents",
    ["degree", "weighted_degree", "betweenness", "pagerank"],
    index=0,
    help="Degree is the required default encoding. Other options help compare different meanings of importance."
)

min_degree = st.sidebar.slider(
    "Minimum degree filter",
    min_value=1,
    max_value=int(nodes_df["degree"].max()),
    value=1,
    step=1,
    help="Increase this to hide minor characters and reduce the hairball problem."
)

min_edge_weight = st.sidebar.slider(
    "Minimum edge weight filter",
    min_value=1,
    max_value=int(edges_df["weight"].max()),
    value=1,
    step=1,
    help="Increase this to show only stronger character co-occurrences."
)

community_options = ["All"] + [str(c) for c in sorted(nodes_df["community"].unique())]
selected_community = st.sidebar.selectbox(
    "Community filter",
    community_options,
    index=0
)

show_labels = st.sidebar.checkbox("Show node labels", value=True)
show_matrix_labels = st.sidebar.checkbox("Show adjacency matrix labels", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Increase the degree or edge-weight filter if the graph becomes too crowded."
)


# --------------------------------------------------
# Filtering logic
# --------------------------------------------------
filtered_nodes = nodes_df[nodes_df["degree"] >= min_degree].copy()

if selected_community != "All":
    filtered_nodes = filtered_nodes[
        filtered_nodes["community"] == int(selected_community)
    ]

H = G.subgraph(filtered_nodes["node"].tolist()).copy()

# Remove weak edges after creating the subgraph.
weak_edges = [
    (u, v)
    for u, v, data in H.edges(data=True)
    if data.get("weight", 1) < min_edge_weight
]
H.remove_edges_from(weak_edges)

# Remove isolated nodes created by the edge-weight filter.
isolated_nodes = list(nx.isolates(H))
H.remove_nodes_from(isolated_nodes)

filtered_metrics_df = nodes_df[nodes_df["node"].isin(H.nodes())].copy()
filtered_edges_df = edges_df[
    edges_df["source"].isin(H.nodes()) & edges_df["target"].isin(H.nodes())
].copy()
filtered_edges_df = filtered_edges_df[filtered_edges_df["weight"] >= min_edge_weight]

if H.number_of_nodes() == 0:
    st.warning("No nodes remain after filtering. Lower the minimum degree or edge-weight threshold.")
    st.stop()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def scale_values(values, min_size=12, max_size=55):
    """Scale a pandas Series to a node-size range."""
    values = pd.Series(values).astype(float)
    if values.max() == values.min():
        return pd.Series([30] * len(values), index=values.index)
    scaled = min_size + (values - values.min()) / (values.max() - values.min()) * (max_size - min_size)
    return scaled


COMMUNITY_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]


def community_color(community_id):
    return COMMUNITY_COLORS[(int(community_id) - 1) % len(COMMUNITY_COLORS)]


def get_positions(graph, layout_name):
    if layout_name == "Circular":
        return nx.circular_layout(graph)
    return nx.spring_layout(graph, seed=42, weight="weight")


def build_pyvis_graph(graph, metrics_df, layout_name, size_col, labels=True):
    """Create an interactive PyVis graph from a NetworkX graph."""
    positions = get_positions(graph, layout_name)

    metric_lookup = metrics_df.set_index("node").to_dict("index")
    sizes = scale_values(metrics_df.set_index("node")[size_col])
    size_lookup = sizes.to_dict()

    net = Network(
        height="720px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="in_line"
    )

    # We compute the layout ourselves so that the graph stays stable.
    net.toggle_physics(False)

    for node in graph.nodes():
        row = metric_lookup[node]
        x, y = positions[node]
        title = (
            f"<b>{node}</b><br>"
            f"Community: {row['community']}<br>"
            f"Degree: {row['degree']}<br>"
            f"Weighted degree: {row['weighted_degree']:.0f}<br>"
            f"Betweenness: {row['betweenness']:.3f}<br>"
            f"PageRank: {row['pagerank']:.3f}"
        )

        net.add_node(
            node,
            label=node if labels else "",
            title=title,
            size=float(size_lookup[node]),
            color=community_color(row["community"]),
            x=float(x * 1100),
            y=float(y * 1100),
        )

    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(
            source,
            target,
            value=float(weight),
            width=float(1 + 0.4 * weight),
            title=f"{source} — {target}<br>Weight: {weight}",
        )

    tmp_path = Path(tempfile.gettempdir()) / "les_mis_network.html"
    net.save_graph(str(tmp_path))
    return tmp_path.read_text(encoding="utf-8")


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Visible nodes", H.number_of_nodes())

with metric_col2:
    st.metric("Visible edges", H.number_of_edges())

with metric_col3:
    st.metric("Communities in full graph", nodes_df["community"].nunique())

with metric_col4:
    st.metric("Layout", layout_choice)


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_graph, tab_matrix, tab_data, tab_explain = st.tabs(
    ["Node-Link Graph", "Adjacency Matrix", "Data Tables", "Explanation"]
)


with tab_graph:
    st.subheader("Interactive Node-Link Visualization")

    st.info(
        "Drag nodes, zoom, and hover over nodes or edges for details. "
        "Use the sidebar filters to reduce clutter and see how the network story changes."
    )

    html = build_pyvis_graph(
        H,
        filtered_metrics_df,
        layout_choice,
        size_metric,
        labels=show_labels
    )
    components.html(html, height=760, scrolling=True)

    st.markdown(
        f"""
        **Current encoding:** node size = `{size_metric}`, node color = detected community,
        and edge thickness = co-occurrence weight. The graph is filtered to characters with
        degree at least **{min_degree}** and edges with weight at least **{min_edge_weight}**.
        """
    )


with tab_matrix:
    st.subheader("Adjacency Matrix View")

    st.write(
        """
        The adjacency matrix shows the same relationships as the node-link graph, but as a table-like heatmap.
        Darker cells mean stronger character co-occurrence. This view is less intuitive for storytelling,
        but it can make dense connection patterns easier to inspect.
        """
    )

    matrix_nodes = filtered_metrics_df.sort_values(["community", "degree"], ascending=[True, False])["node"].tolist()
    A = nx.to_numpy_array(H, nodelist=matrix_nodes, weight="weight")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(A)
    ax.set_title("Weighted Adjacency Matrix")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Characters")

    if show_matrix_labels and len(matrix_nodes) <= 35:
        ax.set_xticks(range(len(matrix_nodes)))
        ax.set_yticks(range(len(matrix_nodes)))
        ax.set_xticklabels(matrix_nodes, rotation=90, fontsize=7)
        ax.set_yticklabels(matrix_nodes, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Edge weight")
    st.pyplot(fig)

    st.caption(
        "If many labels are hidden, turn on matrix labels in the sidebar after filtering the graph to fewer nodes."
    )


with tab_data:
    st.subheader("Node Metrics")
    st.dataframe(filtered_metrics_df.sort_values(size_metric, ascending=False), use_container_width=True)

    node_csv = filtered_metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download visible node metrics",
        data=node_csv,
        file_name="les_mis_visible_node_metrics.csv",
        mime="text/csv"
    )

    st.subheader("Visible Edges")
    st.dataframe(filtered_edges_df, use_container_width=True)

    edge_csv = filtered_edges_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download visible edges",
        data=edge_csv,
        file_name="les_mis_visible_edges.csv",
        mime="text/csv"
    )


with tab_explain:
    st.subheader("What this network represents")
    st.write(
        """
        This network represents character co-occurrence relationships in *Les Misérables*.
        Each node is a character. An edge connects two characters when they appear together,
        and the edge weight represents how strong or frequent that co-occurrence relationship is.
        """
    )

    st.subheader("Why these encodings were chosen")
    st.write(
        """
        I used node size to represent structural importance. By default, size represents degree,
        so larger nodes are characters connected to more other characters. I used node color to
        represent communities detected with greedy modularity, which helps reveal groups of
        characters that interact more strongly with each other than with the rest of the network.
        Edge thickness represents weight, so stronger character relationships appear thicker.
        """
    )

    st.subheader("What the interaction reveals")
    st.write(
        """
        The layout selector changes the story. The force-directed layout is better for seeing clusters,
        bridge characters, and community structure. The circular layout is cleaner and more organized,
        but it can hide the natural clustering of the network. The degree and edge-weight filters help
        reduce the hairball problem by hiding minor characters or weak relationships.
        """
    )

    st.subheader("One limitation")
    st.write(
        """
        A node-link graph can become visually cluttered when many characters and relationships are shown at once.
        The force-directed layout can also mislead viewers because distance is produced by a layout algorithm;
        it is not a measured variable like time, geography, or similarity. Filtering makes the graph readable,
        but it can also hide minor characters who may still matter in the story.
        """
    )

    st.subheader("How this satisfies the assignment")
    st.markdown(
        """
        - **Dataset:** built-in NetworkX Les Misérables character network.
        - **Node-link visualization:** interactive PyVis graph.
        - **Structural encodings:** node size = centrality/degree, node color = community, edge width = weight.
        - **Meaningful widgets:** layout selector, centrality/size selector, degree filter, edge-weight filter, community filter, label checkbox.
        - **Second view:** weighted adjacency matrix.
        - **Limitation:** hairball problem and layout interpretation risk.
        """
    )
