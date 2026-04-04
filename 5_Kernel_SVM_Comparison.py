import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils import generate_data, make_meshgrid, render_shared_controls, init_session_state, show_guided_explanation, show_key_takeaway

st.title("Kernel SVM Comparison")

st.write("Compare a linear SVM with an RBF-kernel SVM on the same 2D dataset.")

init_session_state()
render_shared_controls(include_mapping=False, location="sidebar")

dataset_type = st.session_state["dataset_type"]
n_samples = st.session_state["n_samples"]
noise = st.session_state["noise"]
random_state = st.session_state["random_state"]

st.sidebar.header("SVM Settings")
C_value = st.sidebar.slider("SVM C", 0.1, 10.0, 1.0, 0.1)
gamma_value = st.sidebar.slider("RBF gamma", 0.1, 10.0, 1.0, 0.1)

X, y = generate_data(dataset_type, n_samples, noise, random_state)

linear_clf = SVC(kernel="linear", C=C_value)
linear_clf.fit(X, y)
linear_acc = linear_clf.score(X, y)

rbf_clf = SVC(kernel="rbf", C=C_value, gamma=gamma_value)
rbf_clf.fit(X, y)
rbf_acc = rbf_clf.score(X, y)

xx, yy = make_meshgrid(X)
grid = np.c_[xx.ravel(), yy.ravel()]
Z_linear = linear_clf.predict(grid).reshape(xx.shape)
Z_rbf = rbf_clf.predict(grid).reshape(xx.shape)

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.contourf(xx, yy, Z_linear, alpha=0.2)
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

    w = linear_clf.coef_[0]
    b = linear_clf.intercept_[0]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    if abs(w[1]) > 1e-2:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(w[0] * x_line + b) / w[1]
        ax1.plot(x_line, y_line, linestyle="--", label="Linear boundary")
    elif abs(w[0]) > 1e-8:
        x_vertical = -b / w[0]
        ax1.axvline(x_vertical, linestyle="--", label="Linear boundary")

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Linear SVM")
    ax1.legend()
    st.pyplot(fig1, use_container_width=False)
    st.metric("Linear SVM accuracy", f"{linear_acc:.2%}")

with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.contourf(xx, yy, Z_rbf, alpha=0.2)
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")
    ax2.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax2.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("RBF Kernel SVM")
    ax2.legend()
    st.pyplot(fig2, use_container_width=False)
    st.metric("RBF SVM accuracy", f"{rbf_acc:.2%}")

st.markdown("---")
st.subheader("Interpretation")

if rbf_acc > linear_acc:
    st.success("The RBF kernel handles non-linear structure better on this dataset.")
else:
    st.info("For this dataset, linear and RBF perform similarly. This often happens on linearly separable data.")

st.warning("Try Circles, Moons, XOR, and Spiral. These are the best datasets for seeing the advantage of the RBF kernel.")

show_guided_explanation(
    "Guided Explanation",
    [
        {
            "kind": "info",
            "heading": "Linear SVM",
            "body": "The linear SVM can only draw a straight boundary."
        },
        {
            "kind": "warning",
            "heading": "RBF Kernel SVM",
            "body": "The RBF kernel allows a flexible non-linear decision boundary."
        },
        {
            "kind": "success",
            "heading": "Interpret the comparison",
            "body": "If the RBF score is higher, the dataset likely has non-linear structure that a straight boundary cannot capture well."
        },
    ]
)

if rbf_acc > linear_acc:
    show_key_takeaway(
        f"On this dataset, the RBF kernel outperformed the linear SVM ({rbf_acc:.2%} vs {linear_acc:.2%})."
    )
else:
    show_key_takeaway(
        "On this dataset, linear and RBF perform similarly. This often happens when the data is already close to linearly separable.",
        kind="info"
    )