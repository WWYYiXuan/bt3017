import streamlit as st

st.set_page_config(page_title="Kernel Trick Learning Studio", layout="wide")

st.title("Kernel Trick Learning Studio")
st.subheader("An interactive learning platform for BT3017")

st.write(
    """
    This platform helps undergraduate computing students understand the **kernel trick**.

    You will learn:
    - why some datasets are not linearly separable in 2D,
    - how feature mapping adds a higher dimension,
    - and why this makes linear classification easier.
    """
)

st.markdown("### Learning Journey")
st.markdown(
    """
    **Explore 2D Data** → see how different 2D datasets behave  
    **Feature Mapping** → try different equations that lift the data  
    **Compare** → check whether a chosen mapping improves separation  
    **Before → After** → step through the transformation process  
    **Kernel SVM Comparison** → compare linear vs RBF SVM  
    **PCA Explorer** → learn dimension reduction  
    **Quiz** → test your understanding
    """
)

st.info("Use the left sidebar to move between pages.")