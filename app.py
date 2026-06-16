from __future__ import annotations

from pathlib import Path
import os
import urllib.request
from PIL import Image
import streamlit as st
import torch

from src.config import DEFAULT_MODELS_DIR
from src.gradcam import GradCAM, overlay_heatmap, preprocess_image
from src.models import get_gradcam_target_layer
from src.utils import get_device
from src.gradcam import load_model_for_gradcam

# Page configuration
st.set_page_config(page_title="Pneumonia Detection | XAI", page_icon="🫁", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .metric-container { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def cached_model(checkpoint_path: str):
    device = get_device()
    
    # Auto-download from GitHub Release if not exists
    if not os.path.exists(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        if filename == "best_resnet50_finetuned.pt":
            st.info("📥 Model weights not found on server. Downloading pretrained ResNet-50 weights from GitHub Releases (94MB)... Please wait.")
            url = "https://github.com/Alonessam/pneumonia-xray-detection/releases/download/v1.0.0/best_resnet50_finetuned.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            try:
                urllib.request.urlretrieve(url, checkpoint_path)
                st.success("✅ Download complete!")
            except Exception as e:
                st.error(f"🚨 Failed to download model weights from GitHub Releases: {e}")
                st.stop()
        else:
            st.error("🚨 Trained model not found!")
            st.warning(f"System is looking for `{checkpoint_path}`. Please upload the file or use the default model.")
            st.stop()

    model, checkpoint = load_model_for_gradcam(checkpoint_path, device)
    return model, checkpoint, device

st.title("🫁 Chest X-Ray Pneumonia Detection")
st.markdown("**Deep Learning and Explainable AI (XAI) Decision Support Prototype**")

with st.sidebar:
    st.header("⚙️ System Settings")
    checkpoint_path = st.text_input(
        "Model File",
        value=str(DEFAULT_MODELS_DIR / "best_resnet50_finetuned.pt"),
        help="Path to the trained model weights file."
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This system is developed using the Kaggle Chest X-Ray (Pneumonia) dataset. 
    It aims to provide decision support to radiologists using the **ResNet-50** 
    architecture and the **Grad-CAM** explainability algorithm.
    """)

# Check model path
default_path = str(DEFAULT_MODELS_DIR / "best_resnet50_finetuned.pt")
is_default = os.path.abspath(checkpoint_path) == os.path.abspath(default_path)

if not Path(checkpoint_path).exists() and not is_default:
    # Fallback to older name if finetuned doesn't exist yet
    fallback_path = str(DEFAULT_MODELS_DIR / "best_resnet50.pt")
    if Path(fallback_path).exists():
        checkpoint_path = fallback_path
    else:
        st.error("🚨 Trained model not found!")
        st.warning(f"System is looking for `{checkpoint_path}` or `best_resnet50.pt`. Make sure you have completed the training step.")
        st.code("python -m src.train --model resnet50 --freeze-backbone --epochs 8")
        st.stop()

model, checkpoint, device = cached_model(checkpoint_path)
class_names = checkpoint["class_names"]

st.markdown("### 📤 Upload Chest X-Ray")
uploaded_file = st.file_uploader("Please upload a chest X-Ray image (JPG/JPEG/PNG) for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file)
input_tensor, display_image = preprocess_image(image, checkpoint["img_size"], device)

with st.spinner("AI is analyzing the image..."):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

    gradcam = GradCAM(model, get_gradcam_target_layer(model, checkpoint["model_name"]))
    heatmap, _, _ = gradcam(input_tensor, class_idx=pred_idx)
    gradcam.close()
    overlay = overlay_heatmap(display_image, heatmap)

st.markdown("---")
st.markdown("### 📊 Analysis Results")

# Calculate probabilities
prob_pneumonia = float(probs[class_names.index("PNEUMONIA")]) if "PNEUMONIA" in class_names else float(probs[1])
prob_normal = float(probs[class_names.index("NORMAL")]) if "NORMAL" in class_names else float(probs[0])
pred_label = class_names[pred_idx]

# Metrics display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="🩺 System Prediction", 
        value=pred_label, 
        delta="High Risk" if pred_label == "PNEUMONIA" else "Normal", 
        delta_color="inverse" if pred_label == "PNEUMONIA" else "normal"
    )
with col2:
    st.metric(label="Pneumonia Probability", value=f"{prob_pneumonia*100:.1f}%")
with col3:
    st.metric(label="Normal Probability", value=f"{prob_normal*100:.1f}%")

# Side-by-side images
st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.image(display_image, caption="Original Uploaded Image", use_container_width=True)
with right:
    st.image(overlay, caption="Grad-CAM Heatmap (Model Focus Areas)", use_container_width=True)

# Medical disclaimers
with st.expander("⚠️ Medical Limitations and Disclaimers (IMPORTANT)", expanded=True):
    st.warning("""
    **1. Not a Definitive Diagnosis:** This system is a decision support prototype and does not replace official medical diagnosis. The final decision always belongs to a certified radiologist.
    
    **2. Grad-CAM is Not a 'Lesion Mask':** The heatmap on the right is not a clinical segmentation mask marking the exact boundaries of pneumonia. It simply highlights **which pixel regions** the model focused on when generating its prediction.
    
    **3. Dataset Limitations:** This model was trained on a pediatric, single-center dataset. It does not guarantee the same accuracy for different age groups (adults), other hospitals, or different X-ray machine specifications (generalization boundary).
    """)
