import streamlit as st
import requests

API_URL = "http://ml-app:8000"

st.set_page_config(page_title="CIFAR-10 App", layout="centered")

st.title("📊 Dashboard")

st.write("Framework: ONNX Runtime")
st.write("Dataset: CIFAR-10")
st.write("Expected accuracy: ~73.85%")

if st.button("Health Check"):
    try:
        r = requests.get(f"{API_URL}/", timeout=10)
        r.raise_for_status()
        st.success("API is online")
        st.json(r.json())
    except Exception as e:
        st.error("Health check failed")
        st.write(e)


st.markdown("""
Use the sidebar to navigate to:

- Predict
""")
