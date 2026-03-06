import streamlit as st
import requests

API_URL = "http://ml-app:8000"

st.set_page_config(
    page_title="CIFAR-10 Recognizer",
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]::before {
            content: "LAB NAVIGATION";
            margin-left: 20px;
            margin-top: 20px;
            font-size: 1.1rem;
            font-weight: bold;
            color: #808495;
        }
        .author-link {
            text-decoration: none;
            color: #ff4b4b;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 CIFAR-10 DASHBOARD")
st.caption("🚀 Created by **Lilit Ajoyan** & **Josefin Lesley** | MLOps Students")
st.divider()

with st.sidebar:
    
    st.header("⚙️ System Status")

    if st.button("Check if AI is Ready"):
        try:
            r = requests.get(f"{API_URL}/", timeout=10)
            r.raise_for_status()
            st.success("The AI is awake and ready!")
        except Exception:
            st.error("The AI is currently resting. Please check the Docker connection.")

    st.divider()

    st.header("📦 About the Model")

    st.markdown("""
**Goal:** Identifying everyday objects
                
**Dataset:** CIFAR-10
                
**Expected accuracy:** ~73.85%
""")

st.markdown("## What is this AI app?")

st.markdown("""
Welcome! This app uses a computer "brain" (a neural network) to look at small pictures and tell you what is inside them. 

### How it works:
1. **The Dataset:** The model was taught using the **CIFAR-10** collection—a famous library of 60,000 tiny images used by scientists worldwide to teach computers how to "see."
2. **The Intelligence:** It doesn't just look for colors; it recognizes shapes and patterns to distinguish between a cat and a dog, or a ship and a truck.
3. **The Speed:** We use optimized technology to make sure you get an answer in the blink of an eye.
""")

st.markdown("### What can it recognize?")
st.markdown("The AI has been trained to identify these 10 specific things:")

col1, col2 = st.columns([1, 0.8])

with col1:
    st.markdown("""
- ✈️ **Airplanes** & 🚗 **Cars**
- 🐦 **Birds** & 🐱 **Cats**
- 🦌 **Deer** & 🐶 **Dogs**
- 🐸 **Frogs** & 🐴 **Horses**
- 🚢 **Ships** & 🚚 **Trucks**
""")

with col2:
    st.image(
       "https://storage.googleapis.com/kaggle-media/competitions/kaggle/3649/media/cifar-10.png",
        caption="Examples of the tiny 32x32 images the AI learned from.",
        use_container_width=True
    )

st.divider()    

st.markdown("### 🎓 Developed by MLOps Students")
st.markdown("""
This project is a hands-on lab focused on the lifecycle of machine learning models. 
Connect with the developers on GitHub:
* **[Lilit Ajoyan](https://github.com/LAjoyan)** — *MLOps Student*
* **[Josefin Lesley](https://github.com/Josefin3647)** — *MLOps Student*
""")

st.info("💡 Curious about how we built the infrastructure? Check out the GitHub links above!")