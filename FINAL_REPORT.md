# PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES - FINAL REPORT

**Project Title:** Pneumonia Detection from Chest X-Ray Images using Deep Learning and Explainable AI (XAI)

**Project Metadata:**
- **Developer/Author:** Samet Furkan Bülüç
- **Domain:** Computer Engineering Portfolio Project

---

## 1. Abstract
This project addresses the automatic diagnosis of pneumonia from chest X-ray images using deep learning. Using the open-source "Chest X-Ray Images (Pneumonia)" dataset, two different deep learning architectures were trained and compared. A custom Convolutional Neural Network (Simple CNN) was developed as a baseline model, and a transfer learning approach using a pretrained ResNet-50 backbone was applied as the state-of-the-art model. Following hyperparameter optimization and fine-tuning, the ResNet-50 model achieved **92.79% Accuracy** and **99.23% Recall** on the independent test set. To ensure clinical trust and prevent the model from operating as a "black box," **Grad-CAM (Explainable AI)** was integrated to visualize the activation heatmaps of the final convolutional layer, highlighting the specific pathological regions driving the classification decisions.

## 2. Introduction
Pneumonia is an inflammatory lung disease that affects the air sacs (alveoli) and can lead to life-threatening complications if not diagnosed early. Chest X-rays are the most common, cost-effective, and fast diagnostic tool, but reading them is subjective and prone to diagnostic errors due to fatigue and overlapping anatomical structures.
By leveraging deep learning models, chest X-rays can be analyzed with high precision. This project aims to develop a transparent, explainable decision support prototype that can act as a reliable "second opinion" for healthcare providers.

## 3. Problem Definition
This project is structured as a **binary classification** problem:
- **Input Data:** Frontal chest X-ray images (JPEG format).
- **Output:** Stethoscope-level classification label (`NORMAL` or `PNEUMONIA`) along with probability scores.
- **Target Audience:** Clinicians, radiologists, and emergency triage applications.

## 4. Literature Review
We researched key publications on deep learning for pneumonia detection:
- **Kermany et al. (2018):** Confirmed that deep learning models can perform at radiologist level. *Strengths:* Large-scale pediatric dataset. *Limitations:* Single-center data limiting generalization to other hospital systems.
- **Rajpurkar et al. (CheXNet, 2017):** Outperformed radiologists using a 121-layer DenseNet on ChestX-ray14. *Strengths:* Large dataset and strong architecture. *Limitations:* Frontal views only and lack of explainability.
- **Yadav & Jadhav (2019):** Compared multiple architectures (VGG16, Inception, CapsNet) showing that transfer learning and data augmentation are highly effective for small datasets.
- **Stephen et al. (2019):** Trained a custom CNN from scratch with data augmentation.
- **Toğaçar et al. (2020):** Solved class imbalance using lightweight SqueezeNet and feature selection.

## 5. Benchmark Table

| Study Name | Year | Dataset | Method | Baseline Method | Evaluation Metric | Key Result | Remarks |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- | :--- |
| Kermany et al. | 2018 | Chest X-Ray (Pediatric) | Transfer Learning (CNN) | Traditional CNN | Accuracy, AUC | Radiologist-level performance | Single-center data; no XAI |
| Rajpurkar et al. | 2017 | ChestX-ray14 | 121-layer DenseNet | Radiologist Panel | F1-Score, AUC | Outperformed radiologists | Frontal views only |
| Yadav & Jadhav | 2019 | Kaggle Pneumonia | VGG16, InceptionV3, CapsNet | ORB + SVM | Accuracy, F1 | 93.91% Accuracy | Extensive comparisons |
| Stephen et al. | 2019 | Kaggle Pneumonia | Data Augmentation + CNN | CNN without Augment. | Accuracy, Recall | 93.7% Accuracy | No transfer learning |
| Toğaçar et al. | 2020 | Kaggle Pneumonia | SqueezeNet + Feature Selection | Standard SqueezeNet | Accuracy, F1 | 96.39% Accuracy | Resolves class imbalance |

## 6. Dataset and Preprocessing
The project uses the **"Chest X-Ray Images (Pneumonia)"** dataset (Kaggle/Mendeley), consisting of pediatric chest X-rays.
- **NORMAL Test Set:** 234 Images
- **PNEUMONIA Test Set:** 390 Images
- **Total Test Set:** 624 Images

### Preprocessing Steps:
1. **Resizing:** Images resized to $224 \times 224$ pixels.
2. **Normalization:** Pixel scaling to `[0, 1]` and normalization using ImageNet mean and std.
3. **Data Augmentation:** Applied random rotations (10°), horizontal flips, color jitter, and random affine transformations during training to prevent overfitting and address class imbalance.

## 7. Methods and Implementation

### 7.1. Baseline Method: Simple CNN
A custom 4-layer CNN with batch normalization, ReLU activation, max pooling, dropout (0.35/0.25), and a dense classifier was trained from scratch.

### 7.2. Advanced Method: ResNet-50 (Transfer Learning)
A pretrained ResNet-50 model was integrated. First, feature extraction layers were frozen (`requires_grad = False`) and the classification head (`model.fc`) was replaced. Subsequently, fine-tuning was performed on the upper layers using a low learning rate (1e-5).

## 8. Results and Model Comparison

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline Simple CNN | 71.80% | 0.6904 | **99.49%** | 0.8151 | 0.9238 |
| **ResNet-50 (Fine-Tuned)** | **92.79%** | **0.9021** | 99.23% | **0.9451** | **0.9837** |

### Confusion Matrix Comparison:
- **Baseline CNN:** High Recall (99.49%) but low Precision (0.6904) due to 174 False Positives.
- **ResNet-50:** Drastically reduced False Positives to 42, while keeping False Negatives at only 3.
*Recall (Sensitivity) is highly critical in medical AI because failing to detect a sick patient (False Negative) has fatal consequences, whereas a False Positive is safely caught in follow-up exams. Both models excel at Recall, with ResNet-50 offering superior accuracy.*

## 9. Original Contribution: Explainable AI (XAI) and Grad-CAM
To move beyond black-box predictions, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was integrated. It computes gradients flowing into the final convolutional layer of the network (`layer4[-1]` for ResNet-50) and weights the activation maps to generate a localized heatmap, which is blended with the original chest X-ray. This exposes the specific anatomical regions driving the network's prediction. A Streamlit web application was developed to allow users to interact with this pipeline.

## 10. Conclusion and Limitations
A high-accuracy, explainable AI prototype was successfully built and tested.
- **Limitations:** The model is trained on a pediatric cohort and might require calibration for adults. Grad-CAM maps represent statistical model focus rather than precise medical segmentation masks. It should be used as a decision support tool, not a standalone diagnostic system.

## 11. References
1. Kermany, D. S. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5).
2. Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *arXiv:1711.05225*.
3. Yadav, S. S. & Jadhav, S. M. (2019). Deep convolutional neural network based medical image classification for disease diagnosis. *Journal of Big Data*, 6(1).
4. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.