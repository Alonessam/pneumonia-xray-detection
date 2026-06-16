# PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES - PROJECT PROPOSAL

**Project Metadata:**
- **Author:** Samet Furkan Bülüç
- **Domain:** Computer Engineering Portfolio Project

---

## 1. Domain and Topic Narrowing
- **Primary Domain:** Medical Computer Vision / Image Classification.
- **Specific Problem:** Automated screening and classification of pediatric chest X-ray scans to detect signs of pneumonia.
- **Motivation:** Erken teşhis (early diagnosis) is critical in pediatrics. Deep learning decision support tools can help screen patient queues in low-resource environments.

## 2. Input and Output Definition
- **Input:** $224 \times 224$ normalized RGB/Grayscale chest X-ray image.
- **Output:** Binary classification label (`NORMAL` or `PNEUMONIA`) with Softmax probability scores.

## 3. Research and Literature Plan
- **Literature Review:** Evaluate deep CNNs, VGG, DenseNet, and ResNet architectures used in clinical imaging over the last five years.
- **Metrics Selection:** Focus on Sensitivity (Recall) to ensure low false negative rates, coupled with ROC-AUC.

## 4. Dataset Selection
- **Data Source:** Chest X-Ray Images (Pneumonia) hosted on Kaggle.
- **Composition:** Train and test splits divided into NORMAL and PNEUMONIA subfolders.

## 5. Methodology and Baseline Models
- **Baseline Model:** Simple 4-layer CNN trained from scratch.
- **Advanced Model:** Pretrained ResNet-50 with a custom classification head, trained using transfer learning (backbone frozen) and fine-tuning.
- **XAI Integration:** Grad-CAM hooks added to the final convolutional layers to render visual support heatmaps.
- **Deployment:** A Streamlit interface to load the model checkpoint and process user X-ray uploads dynamically.