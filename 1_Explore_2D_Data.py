import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_data, train_linear_svm, make_meshgrid, render_shared_controls, init_session_state, show_guided_explanation, show_key_takeaway

st.title("Explore 2D Data")

st.write("This page shows a 2D dataset and a linear classifier trying to separate it.")

init_session_state()
render_shared_controls(include_mapping=False, location="sidebar")

dataset_type = st.session_state["dataset_type"]
n_samples = st.session_state["n_samples"]
noise = st.session_state["noise"]
random_state = st.session_state["random_state"]

X, y = generate_data(dataset_type, n_samples, noise, random_state)
clf, acc = train_linear_svm(X, y)

xx, yy = make_meshgrid(X)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(4, 4))
ax.contourf(xx, yy, Z, alpha=0.2)
ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

w = clf.coef_[0]
b = clf.intercept_[0]
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

if abs(w[1]) > 1e-2:
    x_line = np.linspace(x_min, x_max, 200)
    y_line = -(w[0] * x_line + b) / w[1]
    ax.plot(x_line, y_line, linestyle="--", label="Linear boundary")
elif abs(w[0]) > 1e-8:
    x_vertical = -b / w[0]
    ax.axvline(x_vertical, linestyle="--", label="Linear boundary")

ax.set_title("Linear Classifier in Original 2D Space")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig, use_container_width=False)
    st.metric("Linear accuracy in 2D", f"{acc:.2%}")

st.markdown("### What is happening?")

show_guided_explanation(
    "Guided Explanation",
    [
        {
            "kind": "info",
            "heading": "Step 1: Observe the shape",
            "body": "Different 2D datasets have different structures. Some can be separated by a straight line, while others cannot."
        },
        {
            "kind": "warning",
            "heading": "Step 2: Check the boundary",
            "body": "The linear classifier can only form a straight decision boundary. This works well for linearly separable data, but struggles on curved structures."
        },
        {
            "kind": "success",
            "heading": "Step 3: Connect to the lesson",
            "body": "If the data is not separable in its current space, we may need a better representation or a more flexible model."
        },
    ]
)

if dataset_type == "Blobs (linear)":
    show_key_takeaway(
        "Blobs are close to linearly separable, so a straight boundary works well."
    )
elif dataset_type == "Circles":
    show_key_takeaway(
        "Circles show why linear classification fails in 2D: the classes need a curved separation."
    )
elif dataset_type == "Moons":
    show_key_takeaway(
        "Moons are a curved-shape example where linear boundaries usually struggle."
    )
elif dataset_type == "XOR":
    show_key_takeaway(
        "XOR is a classic example where a simple straight-line classifier is not enough."
    )
elif dataset_type == "Spiral":
    show_key_takeaway(
        "Spiral data is highly non-linear, so it strongly motivates non-linear methods or better feature representations.",
        kind="warning"
    )