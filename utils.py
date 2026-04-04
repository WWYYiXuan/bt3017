from sklearn.svm import SVC
import streamlit as st
from sklearn.datasets import make_circles, make_moons, make_blobs
import numpy as np

DEFAULT_STATE = {
    "dataset_type": "Circles",
    "n_samples": 400,
    "noise": 0.08,
    "random_state": 42,
    "mapping_name": "x^2 + y^2",
}

DATASET_OPTIONS = ["Blobs (linear)", "Circles", "Moons", "XOR", "Spiral"]
MAPPING_OPTIONS = ["x^2 + y^2", "x^2 - y^2", "x * y", "|x| + |y|", "sin(x) + sin(y)"]

def init_session_state():
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _copy_widget_to_state(state_key, widget_key):
    st.session_state[state_key] = st.session_state[widget_key]

def render_shared_controls(include_mapping=False, location="sidebar"):
    init_session_state()

    container = st.sidebar if location == "sidebar" else st

    container.header("Dataset Controls")

    # Dataset type
    dataset_widget_key = "_dataset_type_widget"
    if dataset_widget_key not in st.session_state:
        st.session_state[dataset_widget_key] = st.session_state["dataset_type"]

    container.selectbox(
        "Choose dataset type",
        DATASET_OPTIONS,
        key=dataset_widget_key,
        index=DATASET_OPTIONS.index(st.session_state["dataset_type"]),
        on_change=_copy_widget_to_state,
        args=("dataset_type", dataset_widget_key),
    )

    # Number of samples
    n_samples_widget_key = "_n_samples_widget"
    if n_samples_widget_key not in st.session_state:
        st.session_state[n_samples_widget_key] = st.session_state["n_samples"]

    container.slider(
        "Number of samples",
        100, 800, 400, 50,
        key=n_samples_widget_key,
        on_change=_copy_widget_to_state,
        args=("n_samples", n_samples_widget_key),
    )

    # Noise
    noise_widget_key = "_noise_widget"
    if noise_widget_key not in st.session_state:
        st.session_state[noise_widget_key] = st.session_state["noise"]

    container.slider(
        "Noise level",
        0.00, 0.30, 0.08, 0.01,
        key=noise_widget_key,
        on_change=_copy_widget_to_state,
        args=("noise", noise_widget_key),
    )

    # Random seed
    random_state_widget_key = "_random_state_widget"
    if random_state_widget_key not in st.session_state:
        st.session_state[random_state_widget_key] = st.session_state["random_state"]

    container.slider(
        "Random seed",
        0, 100, 42, 1,
        key=random_state_widget_key,
        on_change=_copy_widget_to_state,
        args=("random_state", random_state_widget_key),
    )

    if include_mapping:
        mapping_widget_key = "_mapping_name_widget"
        if mapping_widget_key not in st.session_state:
            st.session_state[mapping_widget_key] = st.session_state["mapping_name"]

        container.selectbox(
            "Choose a mapping equation",
            MAPPING_OPTIONS,
            key=mapping_widget_key,
            index=MAPPING_OPTIONS.index(st.session_state["mapping_name"]),
            on_change=_copy_widget_to_state,
            args=("mapping_name", mapping_widget_key),
        )

    if container.button("Reset to defaults"):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value
        st.session_state["_dataset_type_widget"] = st.session_state["dataset_type"]
        st.session_state["_n_samples_widget"] = st.session_state["n_samples"]
        st.session_state["_noise_widget"] = st.session_state["noise"]
        st.session_state["_random_state_widget"] = st.session_state["random_state"]
        st.session_state["_mapping_name_widget"] = st.session_state["mapping_name"]
        st.rerun()

def generate_data(dataset_type, n_samples, noise, random_state):
    np.random.seed(random_state)

    if dataset_type == "Blobs (linear)":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=2,
            cluster_std=1.2,
            random_state=random_state
        )

    elif dataset_type == "Circles":
        X, y = make_circles(
            n_samples=n_samples,
            factor=0.45,
            noise=noise,
            random_state=random_state
        )

    elif dataset_type == "Moons":
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state
        )

    elif dataset_type == "XOR":
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)

    elif dataset_type == "Spiral":
        n = n_samples // 2
        theta = np.linspace(0, 4 * np.pi, n)
        r = theta

        x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        x2 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])

        X = np.vstack([x1, x2])
        y = np.array([0] * n + [1] * n)

        X = X + noise * np.random.randn(*X.shape)

    else:
        X, y = make_circles(
            n_samples=n_samples,
            factor=0.45,
            noise=noise,
            random_state=random_state
        )

    return X, y

def train_linear_svm(X, y):
    clf = SVC(kernel="linear")
    clf.fit(X, y)
    acc = clf.score(X, y)
    return clf, acc

def make_meshgrid(X, padding=0.5, num_points=300):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num_points),
        np.linspace(y_min, y_max, num_points)
    )
    return xx, yy

def apply_mapping(X, mapping_name):
    x = X[:, 0]
    y = X[:, 1]

    if mapping_name == "x^2 + y^2":
        z = x**2 + y**2
        formula_text = "x² + y²"
    elif mapping_name == "x^2 - y^2":
        z = x**2 - y**2
        formula_text = "x² - y²"
    elif mapping_name == "x * y":
        z = x * y
        formula_text = "x·y"
    elif mapping_name == "|x| + |y|":
        z = np.abs(x) + np.abs(y)
        formula_text = "|x| + |y|"
    elif mapping_name == "sin(x) + sin(y)":
        z = np.sin(x) + np.sin(y)
        formula_text = "sin(x) + sin(y)"
    else:
        z = x**2 + y**2
        formula_text = "x² + y²"

    X_mapped = np.column_stack([x, y, z])
    return z, X_mapped, formula_text

def show_guided_explanation(title, steps):
    st.markdown("---")
    st.subheader(title)

    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        kind = step.get("kind", "info")
        heading = step.get("heading", "")
        body = step.get("body", "")
        text = f"**{heading}**\n\n{body}"

        with col:
            if kind == "success":
                st.success(text)
            elif kind == "warning":
                st.warning(text)
            else:
                st.info(text)

def show_key_takeaway(text, kind="success"):
    st.markdown("### Key Takeaway")
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.info(text)