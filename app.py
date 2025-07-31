import streamlit as st
import os
import time
from PIL import Image
from predict import predict
from featuremap_vis import visualize_feature_maps
from landmark_symmetry import analyze as landmark_analyze
from fft_analyze import fft_spectrum
import shutil
import io
import contextlib

port = os.environ.get("PORT", 8501)

# Page setup
st.set_page_config(
    page_title="Deepfake CyberDetector",
    page_icon="🕵️",
    layout="centered",
)


# Capture stdout to Streamlit
@contextlib.contextmanager
def capture_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


# CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0f1117;
        color: #33ff66;
        font-family: 'Courier New', monospace;
    
    }
    
    /* Main background */
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #111111;  /* Dark gray */
    }
    h1, h2, h3 {
        text-align: center;
        color: #00ffcc;
        text-shadow: 0 0 5px #00ffcc;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: #000;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
    }
    .stFileUploader>div>div>div>div {
        color: #00ffcc;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        background-color: #1c1e26;
        border-left: 4px solid #33ff66;
        box-shadow: 0 0 10px #00ffcc;
    }
    .footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
    }
    .fft-info {
        background-color: #b8c8db;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 3px solid #00ccff;
    }
    .fft-info h4 {
        color: #00ccff;
        margin-top: 0;
    }
    .symmetry-info {
        background-color: #b8c8db;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 3px solid #ffcc00;
    }
    .symmetry-info h4 {
        color: #ffcc00;
        margin-top: 0;
    }
    .symmetry-table {
        width: 100%;
        margin: 10px 0;
        border: 1px solid #33ff66;
        border-collapse: collapse;
    }
    .symmetry-table th, .symmetry-table td {
        padding: 8px;
        text-align: left;
        border: 1px solid #33ff66;
    }
    .symmetry-table tr:nth-child(even) {
        background-color: #b8c8db;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🕵️ Deepfake CyberDetector")
st.markdown("""
Upload an image to detect if it's real or AI-generated. Explore explainability modules: Grad-CAM, Region-Wise Heat Map.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# On upload
if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", "uploaded.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🧠 Preparing neural network..."):
        progress = st.progress(0)
        for i in range(5):
            time.sleep(0.2)
            progress.progress((i + 1) * 20)

        try:
            with capture_output() as buffer:
                result = predict(image_path, save_annotated=False)
            prediction_output = buffer.getvalue()
            progress.empty()

            label = "Real" if "Real" in result else "Fake"
            confidence = result.split("(")[1].split("%")[0]

            st.markdown("##  Analysis Result")
            if label == "Real":
                st.markdown(f"""<div class='result-box'>
                     <strong>REAL FACE DETECTED</strong><br>
                    Confidence: <span style='color:#00ffcc;'>{confidence}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='result-box'>
                     <strong>DEEPFAKE DETECTED</strong><br>
                    Confidence: <span style='color:#ff4c4c;'>{confidence}%</span>
                </div>""", unsafe_allow_html=True)

            st.code(prediction_output, language='text')

            sidebyside_filename = f"sidebyside_{os.path.basename(image_path)}"
            sidebyside_path = os.path.join("output", sidebyside_filename)
            if os.path.exists(sidebyside_path):
                st.image(sidebyside_path, caption="🧠 Grad-CAM & Landmark Visualization", use_container_width=True)
            else:
                st.warning("⚠️ Combined visualization not found.")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

    with st.expander("🧬 Visualize CNN Feature Maps"):
        if st.button("Generate Feature Maps"):
            with st.spinner("🔍 Generating feature maps..."):
                with capture_output() as buffer:
                    visualize_feature_maps(image_path)
                fmap_output = buffer.getvalue()
                for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
                    fmap_path = f"output/featuremap_{layer}.png"
                    if os.path.exists(fmap_path):
                        st.image(fmap_path, caption=f"Feature Map - {layer}", use_container_width=True)
                    else:
                        st.warning(f"❌ Feature map for {layer} not found.")
                #st.code(fmap_output, language='text')



# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
🔐 Deepfake Detection System | ResNet18 + Landmarks + Grad-CAM + Region Wise Attention
</div>
""", unsafe_allow_html=True)