import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import streamlit.components.v1 as components

# ==========================================
# SETUP
# ==========================================

# Page settings
st.set_page_config(page_title="Brain Tumor Check", page_icon="🧠", layout="wide")

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #1A1A1A !important;
        color: #ffffff !important;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 25px;
    }
    .section {
        background-color: #262626;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .result-text {
        font-size: 28px;
        font-weight: 600;
        margin-top: 20px;
    }
    .confidence-bar {
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING
# ==========================================

# 1. Update Labels to match Binary Classification (Alphabetical Order)
labels = ['Healthy', 'Tumor']

# 2. Load the model
try:
    # Ensure this matches the filename saved by your binary training script
    model_path = "Brain_Tumor_Binary.h5" 
    model = load_model(model_path, compile=False)
except IOError:
    st.error("⚠️ Model file not found! Please ensure 'Brain_Tumor_Binary.h5' is in the folder.")
    st.stop()

# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_image(img):
    # 1. Resize to 64x64 (Matches your binary training code)
    img = img.resize((64, 64))
    
    # 2. Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # 3. Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    # 4. Expand dims to create batch (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ==========================================
# UI LAYOUT
# ==========================================

spacer1, left_col, mid_spacer, right_col, spacer2 = st.columns([0.5, 2.5, 0.5, 3, 0.5])

# LEFT COLUMN
with left_col:
    st.markdown('<div class="main-title">🧠 Brain Tumor Check</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an MRI scan to detect if a Tumor is present.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded MRI Image", use_column_width=True)

        if st.button("🔬 Analyze Scan"):
            with st.spinner("Processing image..."):
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)[0]
                class_index = np.argmax(prediction)
                confidence = prediction[class_index] * 100

            result_label = labels[class_index]
            
            # Color logic: Green for Healthy, Red for Tumor
            if result_label == "Healthy":
                result_color = "#00ff88" # Green
                message = "Analysis suggests the brain is Healthy."
            else:
                result_color = "#ff4d4d" # Red
                message = "Analysis detected signs of a Tumor."

            st.markdown(f'<div class="result-text" style="color: {result_color}">Result: {result_label}</div>', unsafe_allow_html=True)
            st.write(f"**{message}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            # Confidence Bars
            st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
            st.write("---")
            for i, label in enumerate(labels):
                score = prediction[i] * 100
                st.write(f"{label}: {score:.1f}%")
                st.progress(int(score))
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN (3D Brain)
with right_col:
    st.markdown("### 🧬 3D Visualization")
    components.html(
        """
        <iframe src='https://my.spline.design/particleaibrain-5f7ifU1Bz1nrsYdrJHdFGYlP/' 
                frameborder='0' 
                width='100%' 
                height='700px' 
                style='border-radius: 12px; background-color: transparent;'>
        </iframe>
        """,
        height=700,
    )

st.markdown("<footer>© 2025 BrainScan AI · Research Tool</footer>", unsafe_allow_html=True)