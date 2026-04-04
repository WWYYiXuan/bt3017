import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import init_session_state, show_guided_explanation, show_key_takeaway

st.title("PCA Explorer")

st.write(
    "PCA solves the opposite problem of feature mapping: instead of increasing dimensions, "
    "it reduces dimensions while preserving as much variation as possible."
)

# Initialize shared app state
init_session_state()

# Shared control from the rest of the app
st.sidebar.header("Shared Control")
st.sidebar.slider("Random seed", 0, 100, 42, 1, key="random_state")

# PCA-specific controls
st.sidebar.header("PCA Controls")
n_points = st.sidebar.slider("Number of points", 50, 400, 150, 10, key="pca_n_points")
angle = st.sidebar.slider("Cloud angle (degrees)", -90, 90, 30, 5, key="pca_angle")
spread_x = st.sidebar.slider("Spread along main direction", 1.0, 6.0, 3.0, 0.1, key="pca_spread_x")
spread_y = st.sidebar.slider("Spread along minor direction", 0.2, 3.0, 0.8, 0.1, key="pca_spread_y")

random_state = st.session_state["random_state"]

np.random.seed(random_state)

theta = np.radians(angle)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

D = np.array([
    [spread_x, 0],
    [0, spread_y]
])

X = np.random.randn(n_points, 2) @ D @ R.T

pca = PCA(n_components=2)
pca.fit(X)
X_center = X.mean(axis=0)

pc1 = pca.components_[0]
pc2 = pca.components_[1]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(X[:, 0], X[:, 1], alpha=0.6, label="Data points")
ax.scatter(X_center[0], X_center[1], color="red", label="Mean")

scale = 3
ax.arrow(
    X_center[0], X_center[1],
    scale * pc1[0], scale * pc1[1],
    head_width=0.15, length_includes_head=True
)
ax.arrow(
    X_center[0], X_center[1],
    scale * pc2[0], scale * pc2[1],
    head_width=0.15, length_includes_head=True
)

ax.text(
    X_center[0] + scale * pc1[0],
    X_center[1] + scale * pc1[1],
    "PC1"
)
ax.text(
    X_center[0] + scale * pc2[0],
    X_center[1] + scale * pc2[1],
    "PC2"
)

ax.set_title("2D Point Cloud with Principal Components")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.axis("equal")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig, use_container_width=False)

col1, col2 = st.columns(2)
with col1:
    st.metric("Explained variance ratio of PC1", f"{pca.explained_variance_ratio_[0]:.2%}")
with col2:
    st.metric("Explained variance ratio of PC2", f"{pca.explained_variance_ratio_[1]:.2%}")

# Optional 1D projection
pca_1d = PCA(n_components=1)
X_1d = pca_1d.fit_transform(X)

st.markdown("---")
st.subheader("Guided Explanation")

box1, box2, box3 = st.columns(3)

with box1:
    st.info(
        "**Step 1: Center and inspect the cloud**\n\n"
        "PCA starts by looking at the overall direction of the data cloud."
    )

with box2:
    st.warning(
        "**Step 2: Find the direction of maximum variance**\n\n"
        "PC1 points along the direction where the data spreads out the most."
    )

with box3:
    st.success(
        "**Step 3: Reduce dimension**\n\n"
        "To compress the data, PCA projects points onto the most important direction(s)."
    )

st.markdown("---")
st.write("### Why PCA belongs in this app")

st.success(
    "Feature mapping and PCA are complementary ideas: feature mapping increases dimensions "
    "to make classification easier, while PCA reduces dimensions to simplify the data."
)

show_guided_explanation(
    "Guided Explanation",
    [
        {
            "kind": "info",
            "heading": "Step 1: Observe the cloud",
            "body": "PCA studies the overall direction and spread of the point cloud."
        },
        {
            "kind": "warning",
            "heading": "Step 2: Find important directions",
            "body": "PC1 points in the direction of maximum variance, while PC2 captures the next most important direction."
        },
        {
            "kind": "success",
            "heading": "Step 3: Reduce dimension",
            "body": "By projecting onto the most important components, PCA keeps useful structure while reducing dimensionality."
        },
    ]
)

show_key_takeaway(
    "Kernel mapping increases dimensions to help separation, while PCA reduces dimensions to simplify representation."
)