import streamlit as st
import requests

API_URL = "http://ml-app:8000"

st.set_page_config(
    page_title="CIFAR-10 App",
    layout="centered"
)

st.title("🏠 Home")

with st.sidebar:
    # st.title("⚙️ System")

    st.header("⚙️ Health")

    if st.button("Health Check"):
        try:
            r = requests.get(f"{API_URL}/", timeout=10)
            r.raise_for_status()
            st.success("API is online!")
            st.json(r.json())
        except Exception as e:
            st.error("Health check failed")
            st.write(e)

    st.divider()

    st.header("📦 Model info")

    st.markdown("""
**Framework:** ONNX Runtime
                
**Dataset:** CIFAR-10
                
**Expected accuracy:** ~73.85%
""")

st.markdown("## CIFAR-10 Image Classifier")

st.markdown("""
This application demonstrates an **image classification model trained on the CIFAR-10 dataset**.

CIFAR-10 is a well-known benchmark dataset in computer vision consisting of **60,000 small color images (32×32 pixels)** divided into **10 different classes**.

The model has been exported to the **ONNX format** and is served through a **FastAPI backend**, allowing efficient inference through **ONNX Runtime**.
""")

st.markdown("### Classes in CIFAR-10")
st.markdown("The model can classify the following objects:")

col1, col2 = st.columns([1, 0.8])

with col1:
    st.markdown("""
- ✈️ Airplane  
- 🚗 Automobile  
- 🐦 Bird  
- 🐱 Cat  
- 🦌 Deer  
- 🐶 Dog  
- 🐸 Frog  
- 🐴 Horse  
- 🚢 Ship  
- 🚚 Truck  
""")

with col2:
    st.image(
        "https://storage.googleapis.com/kaggle-media/competitions/kaggle/3649/media/cifar-10.png",
        use_container_width=True
    )

st.markdown("""
Use the **sidebar** to navigate to the prediction page where you can upload an image and test the model.

➡️ Go to **Predict** to try it out!
""")