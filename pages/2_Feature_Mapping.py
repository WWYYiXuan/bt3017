import streamlit as st
import pandas as pd
import plotly.express as px
from utils import generate_data, apply_mapping, render_shared_controls, init_session_state, show_guided_explanation, show_key_takeaway

st.title("Feature Mapping Explorer")

st.write("Choose a feature-mapping equation and observe how the 2D data is lifted into 3D.")

init_session_state()
render_shared_controls(include_mapping=True, location="sidebar")

dataset_type = st.session_state["dataset_type"]
n_samples = st.session_state["n_samples"]
noise = st.session_state["noise"]
random_state = st.session_state["random_state"]
mapping_name = st.session_state["mapping_name"]

X, y = generate_data(dataset_type, n_samples, noise, random_state)
z, X_mapped, formula_text = apply_mapping(X, mapping_name)

df = pd.DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "z": z,
    "class": y.astype(str)
})

fig = px.scatter_3d(
    df,
    x="x1",
    y="x2",
    z="z",
    color="class",
    title=f"3D Mapped Data: z = {formula_text}",
    labels={"class": "Class"}
)

st.plotly_chart(fig, use_container_width=True)
st.info(f"Current mapping: z = {formula_text}")

show_guided_explanation(
    "Guided Explanation",
    [
        {
            "kind": "info",
            "heading": "Step 1: Start from 2D",
            "body": "The original data only has x and y coordinates."
        },
        {
            "kind": "warning",
            "heading": "Step 2: Add a new feature",
            "body": f"The mapping z = {formula_text} creates a new representation of the same data."
        },
        {
            "kind": "success",
            "heading": "Step 3: Why this matters",
            "body": "A useful mapping can spread the classes apart in higher-dimensional space, making linear separation easier."
        },
    ]
)

show_key_takeaway(
    f"Feature mapping changes the representation of the data, not the labels. Here, the active mapping is z = {formula_text}."
)