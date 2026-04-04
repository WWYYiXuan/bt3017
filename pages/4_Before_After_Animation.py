import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from utils import generate_data, train_linear_svm, apply_mapping, make_meshgrid, render_shared_controls, init_session_state, show_guided_explanation, show_key_takeaway

st.title("Before → After Transformation")

st.write("Move through the stages to see how feature mapping changes the classification problem.")

init_session_state()
render_shared_controls(include_mapping=True, location="sidebar")

dataset_type = st.session_state["dataset_type"]
n_samples = st.session_state["n_samples"]
noise = st.session_state["noise"]
random_state = st.session_state["random_state"]
mapping_name = st.session_state["mapping_name"]

stage = st.slider("Stage", 0, 3, 0)
stage_labels = {
    0: "Stage 0: Original data",
    1: "Stage 1: Linear boundary in 2D",
    2: "Stage 2: Data after feature mapping",
    3: "Stage 3: Linear classifier after mapping"
}
st.subheader(stage_labels[stage])

X, y = generate_data(dataset_type, n_samples, noise, random_state)

if stage in [0, 1]:
    clf_2d, acc_2d = train_linear_svm(X, y)
    xx, yy = make_meshgrid(X)
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))

    if stage == 1:
        ax.contourf(xx, yy, Z, alpha=0.2)

    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

    if stage == 1:
        w = clf_2d.coef_[0]
        b = clf_2d.intercept_[0]
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        if abs(w[1]) > 1e-2:
            x_line = np.linspace(x_min, x_max, 200)
            y_line = -(w[0] * x_line + b) / w[1]
            ax.plot(x_line, y_line, linestyle="--", label="Linear boundary")
        elif abs(w[0]) > 1e-8:
            x_vertical = -b / w[0]
            ax.axvline(x_vertical, linestyle="--", label="Linear boundary")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax.set_title("2D View")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig, use_container_width=False)

else:
    z, X_mapped, formula_text = apply_mapping(X, mapping_name)

    df_3d = pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "z": z,
        "class": y.astype(str)
    })

    fig3d = px.scatter_3d(
        df_3d,
        x="x1",
        y="x2",
        z="z",
        color="class",
        title=f"Mapped data: z = {formula_text}",
        labels={"class": "Class"}
    )
    st.plotly_chart(fig3d, use_container_width=True)

    clf_3d, acc_3d = train_linear_svm(X_mapped, y)

    if stage == 2:
        st.info("The points are now represented in a higher-dimensional space.")
    else:
        st.success(f"Linear accuracy after mapping: {acc_3d:.2%}")

if stage == 0:
    show_guided_explanation(
        "Guided Explanation",
        [
            {"kind": "info", "heading": "Stage 0", "body": "Start by looking only at the original shape of the dataset."},
            {"kind": "warning", "heading": "Why it matters", "body": "The geometry of the data tells us whether a straight line is likely to work."},
            {"kind": "success", "heading": "Next step", "body": "Move to Stage 1 to see the actual linear boundary."},
        ]
    )
    show_key_takeaway("Always inspect the data structure before choosing a model.")
elif stage == 1:
    show_guided_explanation(
        "Guided Explanation",
        [
            {"kind": "info", "heading": "Stage 1", "body": "Now the linear boundary is visible in the original 2D space."},
            {"kind": "warning", "heading": "What to notice", "body": "If the shape is curved or nested, the straight boundary may fail."},
            {"kind": "success", "heading": "Next step", "body": "Move to Stage 2 to see the higher-dimensional representation."},
        ]
    )
    show_key_takeaway("Linear models are limited by the space in which the data is represented.")
elif stage == 2:
    show_guided_explanation(
        "Guided Explanation",
        [
            {"kind": "info", "heading": "Stage 2", "body": "The same points now live in a higher-dimensional feature space."},
            {"kind": "warning", "heading": "What changed", "body": "The labels stayed the same, but the representation changed."},
            {"kind": "success", "heading": "Why useful", "body": "This new space may separate classes more clearly."},
        ]
    )
    show_key_takeaway("Feature mapping helps by reshaping the data geometry.")
else:
    show_guided_explanation(
        "Guided Explanation",
        [
            {"kind": "info", "heading": "Stage 3", "body": "Now evaluate how the linear classifier behaves after mapping."},
            {"kind": "warning", "heading": "Compare carefully", "body": "The key question is whether the new representation improved separability."},
            {"kind": "success", "heading": "Concept link", "body": "This is the core intuition behind the kernel trick."},
        ]
    )
    show_key_takeaway("A good feature mapping can turn a hard non-linear problem into an easier linear one.")