import streamlit as st
import random

st.title("Mini Test: Kernel Trick & PCA")
st.write("Answer the questions below and check each one. A random set of 5 questions is shown each time.")

QUESTION_BANK = [
    {
        "question": "Why does a linear classifier struggle on the original circular dataset?",
        "options": [
            "Because there are too many samples",
            "Because the classes are arranged in concentric circles and cannot be separated well by a straight line",
            "Because the data has three dimensions",
            "Because the labels are missing"
        ],
        "answer": "Because the classes are arranged in concentric circles and cannot be separated well by a straight line",
        "explanation": "Circular classes need a curved boundary. A straight line cannot separate the inner and outer rings well."
    },
    {
        "question": "What does z = x1² + x2² do?",
        "options": [
            "It removes all noise",
            "It changes the labels",
            "It adds a new feature that lifts points into higher-dimensional space",
            "It converts the task into regression"
        ],
        "answer": "It adds a new feature that lifts points into higher-dimensional space",
        "explanation": "The mapping adds a new coordinate based on distance from the origin, which can make separation easier."
    },
    {
        "question": "Why is the transformed space useful?",
        "options": [
            "It reduces the number of samples",
            "It makes the data easier to separate with a linear classifier",
            "It removes the need for labels",
            "It only improves visual appearance"
        ],
        "answer": "It makes the data easier to separate with a linear classifier",
        "explanation": "The transformed representation can turn a non-linear problem into one that is easier for a linear classifier."
    },
    {
        "question": "What is the main goal of PCA?",
        "options": [
            "Increase the number of features",
            "Reduce dimensionality while preserving important variation",
            "Change class labels",
            "Generate new data points"
        ],
        "answer": "Reduce dimensionality while preserving important variation",
        "explanation": "PCA projects data onto principal components so fewer dimensions can still retain most of the structure."
    },
    {
        "question": "What does the first principal component (PC1) represent?",
        "options": [
            "The direction of minimum variance",
            "The mean of the dataset",
            "The direction of maximum variance",
            "A random axis"
        ],
        "answer": "The direction of maximum variance",
        "explanation": "PC1 is the direction along which the data varies the most."
    },
    {
        "question": "Which dataset is a classic example of a non-linearly separable problem?",
        "options": [
            "Blobs (linear)",
            "XOR",
            "Separated Gaussian clusters",
            "A single point"
        ],
        "answer": "XOR",
        "explanation": "XOR cannot be separated by one straight line, so it is a classic example for non-linear methods."
    },
    {
        "question": "Why can RBF SVM outperform linear SVM on moons or circles?",
        "options": [
            "Because it removes labels",
            "Because it creates flexible non-linear decision boundaries",
            "Because it always uses fewer samples",
            "Because it converts the task into PCA"
        ],
        "answer": "Because it creates flexible non-linear decision boundaries",
        "explanation": "RBF kernels let the classifier model curved boundaries that fit non-linear data better."
    },
    {
        "question": "Why is centering important in PCA?",
        "options": [
            "It changes labels",
            "It avoids the mean shifting the principal direction",
            "It increases dimensions",
            "It sorts the samples"
        ],
        "answer": "It avoids the mean shifting the principal direction",
        "explanation": "PCA should measure variance around the mean, not around the origin unless the data is already centered."
    },
    {
        "question": "What does feature mapping change?",
        "options": [
            "The class labels",
            "The representation of the data",
            "The number of classes",
            "The evaluation metric"
        ],
        "answer": "The representation of the data",
        "explanation": "Feature mapping changes how the same data points are represented, not their labels."
    },
    {
        "question": "In your app, what is the key difference between kernel mapping and PCA?",
        "options": [
            "Kernel mapping reduces dimension while PCA increases it",
            "Kernel mapping increases dimension, while PCA reduces dimension",
            "They are exactly the same",
            "Both are only for visualization"
        ],
        "answer": "Kernel mapping increases dimension, while PCA reduces dimension",
        "explanation": "Kernel-style mapping adds useful features, while PCA compresses data into fewer dimensions."
    },
]

# ---------- Session state setup ----------
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = random.sample(QUESTION_BANK, 5)

if "quiz_checked" not in st.session_state:
    st.session_state.quiz_checked = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

def reset_quiz():
    st.session_state.quiz_questions = random.sample(QUESTION_BANK, 5)
    st.session_state.quiz_checked = {}
    st.session_state.quiz_submitted = False
    for i in range(5):
        answer_key = f"quiz_answer_{i}"
        if answer_key in st.session_state:
            del st.session_state[answer_key]

st.button("New Test", on_click=reset_quiz)

st.markdown("---")

# ---------- Questions ----------
for i, q in enumerate(st.session_state.quiz_questions):
    st.markdown(f"### Q{i+1}. {q['question']}")

    answer_key = f"quiz_answer_{i}"
    checked_key = f"quiz_checked_{i}"

    selected = st.radio(
        "Choose one answer:",
        q["options"],
        key=answer_key,
        label_visibility="collapsed"
    )

    def mark_checked(idx=i):
        st.session_state.quiz_checked[idx] = True

    st.button(f"Check Question {i+1}", key=checked_key, on_click=mark_checked)

    if st.session_state.quiz_checked.get(i, False):
        if st.session_state[answer_key] == q["answer"]:
            st.success("Correct.")
            st.info(f"Explanation: {q['explanation']}")
        else:
            st.error(f"Incorrect. Correct answer: {q['answer']}")
            st.info(f"Explanation: {q['explanation']}")

    st.markdown("---")

# ---------- Final score ----------
if st.button("Submit Entire Test"):
    st.session_state.quiz_submitted = True

if st.session_state.quiz_submitted:
    score = 0
    for i, q in enumerate(st.session_state.quiz_questions):
        answer_key = f"quiz_answer_{i}"
        if answer_key in st.session_state and st.session_state[answer_key] == q["answer"]:
            score += 1

    st.subheader("Final Score")
    st.metric("Score", f"{score} / 5")
    st.progress(score / 5)

    if score == 5:
        st.success("Excellent work.")
    elif score >= 3:
        st.info("Good job. You understand most of the concepts.")
    else:
        st.warning("Review the lesson pages and try again.")