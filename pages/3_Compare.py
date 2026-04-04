import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
from utils import generate_data, train_linear_svm, apply_mapping, make_meshgrid, render_shared_controls, init_session_state, show_guided_explanation, show_key_takeaway

st.title("Compare Before vs After")

init_session_state()
render_shared_controls(include_mapping=True, location="sidebar")

dataset_type = st.session_state["dataset_type"]
n_samples = st.session_state["n_samples"]
noise = st.session_state["noise"]
random_state = st.session_state["random_state"]
mapping_name = st.session_state["mapping_name"]

X, y = generate_data(dataset_type, n_samples, noise, random_state)

clf_2d, acc_2d = train_linear_svm(X, y)
xx, yy = make_meshgrid(X)
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig2d, ax2d = plt.subplots(figsize=(5, 5))
ax2d.contourf(xx, yy, Z, alpha=0.2)
ax2d.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
ax2d.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

w = clf_2d.coef_[0]
b = clf_2d.intercept_[0]
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

if abs(w[1]) > 1e-2:
    x_line = np.linspace(x_min, x_max, 200)
    y_line = -(w[0] * x_line + b) / w[1]
    ax2d.plot(x_line, y_line, linestyle="--", label="Linear boundary")
elif abs(w[0]) > 1e-8:
    x_vertical = -b / w[0]
    ax2d.axvline(x_vertical, linestyle="--", label="Linear boundary")

ax2d.set_xlim(x_min, x_max)
ax2d.set_ylim(y_min, y_max)
ax2d.set_aspect("equal", adjustable="box")
ax2d.set_title("Before: Original 2D Space")
ax2d.legend()

z, X_mapped, formula_text = apply_mapping(X, mapping_name)
clf_3d, acc_3d = train_linear_svm(X_mapped, y)

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
    labels={"class": "Class"}
)

fig3d.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=40, b=0)
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Before: Original 2D")
    st.pyplot(fig2d, use_container_width=True)
    st.metric("Linear accuracy in 2D", f"{acc_2d:.2%}")

with col2:
    st.markdown(f"#### After: z = {formula_text}")
    st.plotly_chart(fig3d, use_container_width=True)
    st.metric("Accuracy after mapping", f"{acc_3d:.2%}")

show_guided_explanation(
    "Guided Explanation",
    [
        {
            "kind": "info",
            "heading": "Step 1: Look at the original space",
            "body": "The left figure shows how a linear classifier behaves on the raw 2D dataset."
        },
        {
            "kind": "warning",
            "heading": "Step 2: Apply mapping",
            "body": f"The right figure shows the same dataset after applying z = {formula_text}."
        },
        {
            "kind": "success",
            "heading": "Step 3: Compare the results",
            "body": "If the mapped-space accuracy improves, the new feature representation is helping the linear classifier."
        },
    ]
)

if acc_3d > acc_2d:
    show_key_takeaway(
        f"For this dataset, z = {formula_text} improved linear separability from {acc_2d:.2%} to {acc_3d:.2%}."
    )
else:
    show_key_takeaway(
        f"For this dataset, z = {formula_text} did not improve much. This shows that not all feature mappings are equally useful.",
        kind="warning"
    )