import streamlit as st
import requests

API_URL = "http://ml-app:8000"

# --- Page setup ---
st.set_page_config(page_title="CIFAR-10 (ONNX) UI", layout="centered")

st.title("CIFAR-10 Classifier (ONNX)")
st.markdown(
"""
Upload an image of a ✈️ fast airplane, 🚗 an expensive car, 🐦 an ugly bird, 
🐱 a smelly cat, 🦌 a sweet deer, 🐶 your neighbours dog, 🐸 a slaimy frog, 
🐴 a fat horse, 🚢 an unsinkable ship or 🚚 a very yellow truck, and see if the model gets it right!
"""
)

# --- Sidebar: system / health check / model info ---
with st.sidebar:
    st.header("🩺 System")

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
    st.write("Framework: **ONNX Runtime**")
    st.write("Dataset: **CIFAR-10**")
    st.write("Expected accuracy: **~73.85%**")

# --- Upload ---
uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"])

st.divider()

# Emoji mapping for nicer output
EMOJIS = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐶",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚚",
}

# --- Main UI: image + prediction side-by-side ---
if uploaded is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Preview")
        st.image(uploaded, caption="Uploaded image", width=250)


    with col2:
        st.write("")  # spacing

    if uploaded is not None:

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
else:
    st.info("👆 Here we go!")