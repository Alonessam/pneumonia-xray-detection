from __future__ import annotations

from pathlib import Path

from PIL import Image
import streamlit as st
import torch

from src.config import DEFAULT_MODELS_DIR
from src.gradcam import GradCAM, overlay_heatmap, preprocess_image
from src.models import get_gradcam_target_layer
from src.utils import get_device
from src.gradcam import load_model_for_gradcam


st.set_page_config(page_title="Zatürre Tespiti | XAI", page_icon="🫁", layout="wide")

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
    model, checkpoint = load_model_for_gradcam(checkpoint_path, device)
    return model, checkpoint, device

st.title("🫁 Göğüs Röntgeninden Zatürre Tespiti")
st.markdown("**Derin Öğrenme ve Açıklanabilir Yapay Zeka (XAI) Destekli Karar Prototipi**")

with st.sidebar:
    st.header("⚙️ Sistem Ayarları")
    checkpoint_path = st.text_input(
        "Model Dosyası",
        value=str(DEFAULT_MODELS_DIR / "best_resnet50_finetuned.pt"),
        help="Eğitilmiş model ağırlıklarının bulunduğu dosya yolu."
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Hakkında")
    st.info("""
    Bu sistem, Kaggle Chest X-Ray (Pneumonia) veri seti kullanılarak geliştirilmiştir. 
    **ResNet-50** mimarisi ve **Grad-CAM** algoritması kullanılarak radyologlara 
    karar destek sağlaması amaçlanmıştır.
    """)

if not Path(checkpoint_path).exists():
    # Fallback to older name if finetuned doesn't exist yet
    if Path(str(DEFAULT_MODELS_DIR / "best_resnet50.pt")).exists():
        checkpoint_path = str(DEFAULT_MODELS_DIR / "best_resnet50.pt")
    else:
        st.error("🚨 Eğitilmiş model bulunamadı!")
        st.warning(f"Sistem şu an `{checkpoint_path}` veya `best_resnet50.pt` dosyasını arıyor. Eğitim adımını tamamladığınızdan emin olun.")
        st.code("python -m src.train --model resnet50 --freeze-backbone --epochs 8")
        st.stop()

model, checkpoint, device = cached_model(checkpoint_path)
class_names = checkpoint["class_names"]

st.markdown("### 📤 Röntgen Yükleme")
uploaded_file = st.file_uploader("Lütfen analiz edilecek göğüs röntgenini (X-Ray) yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file)
input_tensor, display_image = preprocess_image(image, checkpoint["img_size"], device)

with st.spinner("Yapay zeka analiz ediyor..."):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

    gradcam = GradCAM(model, get_gradcam_target_layer(model, checkpoint["model_name"]))
    heatmap, _, _ = gradcam(input_tensor, class_idx=pred_idx)
    gradcam.close()
    overlay = overlay_heatmap(display_image, heatmap)

st.markdown("---")
st.markdown("### 📊 Analiz Sonuçları")

# Olasılıkları hesapla
prob_pneumonia = float(probs[class_names.index("PNEUMONIA")]) if "PNEUMONIA" in class_names else float(probs[1])
prob_normal = float(probs[class_names.index("NORMAL")]) if "NORMAL" in class_names else float(probs[0])
pred_label = class_names[pred_idx]

# Üst metrikler
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🩺 Sistem Tahmini", value=pred_label, delta="Yüksek Risk" if pred_label=="PNEUMONIA" else "Normal Risk", delta_color="inverse" if pred_label=="PNEUMONIA" else "normal")
with col2:
    st.metric(label="Zatürre Olasılığı", value=f"%{prob_pneumonia*100:.1f}")
with col3:
    st.metric(label="Normal Olasılığı", value=f"%{prob_normal*100:.1f}")

# Görseller
st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.image(display_image, caption="Orijinal Yüklenen Görüntü", use_container_width=True)
with right:
    st.image(overlay, caption="Grad-CAM Isı Haritası (Odaklanılan Bölgeler)", use_container_width=True)

# Uyarı ve açıklamalar
with st.expander("⚠️ Tıbbi Sınırlılıklar ve Açıklamalar (ÖNEMLİ)", expanded=True):
    st.warning("""
    **1. Kesin Tanı Değildir:** Bu sistem bir karar destek prototipidir ve kesin tıbbi tanı yerine geçmez. Nihai karar her zaman uzman bir radyoloğa aittir.
    
    **2. Grad-CAM Bir 'Maske' Değildir:** Sağdaki ısı haritası, modelin "zatürreli bölge kesin burasıdır" dediği tıbbi bir maske değildir. Sadece modelin tahmini yaparken ağırlıklı olarak **hangi piksellere odaklandığını** gösterir.
    
    **3. Veri Seti Sınırlılığı:** Bu model pediatrik ve tek merkezli bir veri seti ile eğitilmiştir. Farklı yaş grupları (yetişkinler), farklı hastaneler veya farklı X-Ray cihazlarından gelen görüntülerde aynı başarı oranını garanti etmez (Genelleme sınırı).
    """)
