import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CIFAR-10 (ONNX) UI", layout="centered")
st.title("CIFAR-10 Classifier (ONNX)")
st.caption("Streamlit frontend that calls the FastAPI backend (/predict).")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Health Check"):
        r = requests.get(f"{API_URL}/", timeout=10)
        st.json(r.json())

with col2:
    st.write("")  # spacing

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        files = {
            "file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")
        }

        with st.spinner("Sending request to FastAPI..."):
            r = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if r.status_code != 200:
            st.error(f"API error {r.status_code}")
            st.text(r.text)
        else:
            data = r.json()
            st.success("Prediction complete!")
            st.write(f"**Label:** {data['label']}")
            st.write(f"**Class ID:** {data['class_id']}")
            st.write(f"**Confidence:** {data['confidence']}")