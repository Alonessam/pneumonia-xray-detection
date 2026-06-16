# Chest X-Ray Pneumonia Detection (with Explainable AI - Grad-CAM)

This repository contains an AI-based decision support prototype that classifies chest X-ray images into `NORMAL` and `PNEUMONIA` classes. Built in alignment with deep learning research methodologies, the project includes a baseline CNN model, a transfer learning ResNet-50 model, performance comparisons, error analysis, Grad-CAM (Explainable AI) visualization, and an interactive Streamlit web application.

---

## 📌 Project Overview

1. **Dataset:** Utilizing the open-source "Chest X-Ray Images (Pneumonia)" dataset from Kaggle/Mendeley.
2. **Preprocessing:** Resizing images to $224 \times 224$ pixels, scaling, applying color jittering, random rotations, and normalizing using ImageNet statistics.
3. **Baseline Model:** Training a custom 4-layer Convolutional Neural Network (Simple CNN) from scratch.
4. **State-of-the-Art Model:** Applying Transfer Learning and Fine-Tuning using a pretrained ResNet-50 backbone.
5. **Class Imbalance Mitigation:** Computing and injecting class weights into the loss function (`CrossEntropyLoss`) to penalize minority class errors.
6. **Evaluation:** Comparing models on an independent test set using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
7. **Explainable AI (XAI):** Implementing Grad-CAM to visualize activation heatmaps of the final convolutional layer, highlighting the pathological regions that drive the model's predictions.
8. **Interactive UI:** Providing a user-friendly Streamlit web interface for physicians to upload chest X-ray images, obtain predictions, and inspect the Grad-CAM focus areas.

---

## 📂 Project Structure

```text
pneumonia_xray_ai/
  app.py                     # Streamlit web application
  requirements.txt           # Main python dependencies
  prepare_data.py            # Data preprocessing utility
  compile_report.py          # Report compilation helper
  FINAL_RAPOR.md             # Clean project markdown report (Turkish)
  Zaturre_Tespiti_Final_Raporu.docx       # Clean Word Final Report (Turkish)
  Zaturre_Tespiti_On_Arastirma_Raporu.docx # Clean Word Pre-Research Report (Turkish)
  Zaturre_Tespiti_Proje_Onerisi.docx       # Clean Word Project Proposal (Turkish)
  data/
    chest_xray/              # (Ignored) Dataset directory containing train/val/test splits
  models/                    # (Ignored) Saved model weights (*.pt)
  reports/                   # Stored training histories, metrics, and summary files
  outputs/                   # Saved Grad-CAM visualization outputs
  src/
    config.py                # Hyperparameters, paths, and normalization constants
    data.py                  # PyTorch Custom Dataset, Transforms, and Dataloaders
    models.py                # SimpleCNN structure and ResNet-50 setup
    train.py                 # PyTorch training pipeline
    evaluate.py              # Performance evaluation and metric computation
    gradcam.py               # Grad-CAM class and heatmap overlays
    utils.py                 # Reproducibility seeds, checkpoint saving, and device check
```

---

## ⚡ Setup and Installation

### 1. Create Virtual Environment and Install Dependencies
Open your terminal inside the project root folder and execute:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure PyTorch with GPU (CUDA) Support (Optional)
If you are running on an NVIDIA GPU (e.g., RTX 3060/4060) and want to utilize CUDA, run:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```
Verify your GPU installation:
```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

---

## 📊 Dataset Preparation

The dataset used is **Chest X-Ray Images (Pneumonia)** by Paul Mooney (Mendeley source). You can download it directly via the Kaggle CLI:
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```
Ensure your directory tree matches:
```text
data/chest_xray/train/NORMAL/
data/chest_xray/train/PNEUMONIA/
data/chest_xray/test/NORMAL/
data/chest_xray/test/PNEUMONIA/
```
Alternatively, download the ZIP archive from Kaggle and extract it manually into `data/chest_xray/`. If you already have `archive.zip` in your downloads, you can run:
```bash
python prepare_data.py
```

---

## 🏋️ Training the Models

### 1. Train Baseline CNN
Trains a simple CNN architecture from scratch for 10 epochs:
```bash
python -m src.train --model simple_cnn --epochs 10 --batch-size 32
```

### 2. Train ResNet-50 (Frozen Backbone)
Trains only the newly added classification head while keeping the feature extraction layers frozen:
```bash
python -m src.train --model resnet50 --freeze-backbone --epochs 8 --batch-size 32
```

### 3. Fine-Tuning ResNet-50
Resumes training from the frozen checkpoint with a smaller learning rate to tune the upper layers:
```bash
python -m src.train --model resnet50 --resume models/best_resnet50_frozen.pt --epochs 5 --batch-size 16 --lr 1e-5
```
Saved models will be stored in the `models/` directory, and training logs will be written to `reports/`.

---

## 📈 Evaluation and Benchmarks

To evaluate the fine-tuned ResNet-50 model on the test dataset:
```bash
python -m src.evaluate --checkpoint models/best_resnet50_finetuned.pt
```

### Summary of Test Set Performance

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Baseline Simple CNN** | 71.80% | 0.6904 | **99.49%** | 0.8151 | 0.9238 |
| **ResNet-50 + Fine-Tuning** | **92.79%** | **0.9021** | 99.23% | **0.9451** | **0.9837** |

*Note: In medical diagnostics, **Recall (Sensitivity)** is the most crucial metric since missing a positive pneumonia patient (False Negative) can be life-threatening. Both models achieve over 99% Recall, with ResNet-50 significantly reducing False Positives (improving Precision and overall Accuracy).*

---

## 🔍 Explainable AI (XAI) with Grad-CAM

To generate a Grad-CAM heatmap for a single image, pointing out exactly where the model is looking:
```bash
python -m src.gradcam --checkpoint models/best_resnet50_finetuned.pt --image data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```
The resulting superimposed heatmap overlay will be saved to the `outputs/` folder.

---

## 💻 Web User Interface

Start the local Streamlit web application:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. The app enables users to:
1. Upload any chest X-ray image (JPG/PNG).
2. View classification probabilities for both categories.
3. Inspect the side-by-side **Grad-CAM Heatmap** overlay, exposing the model's focus points.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).