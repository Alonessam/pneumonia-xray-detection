# PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES - PRE-RESEARCH REPORT

**Project Metadata:**
- **Author:** Samet Furkan Bülüç
- **Domain:** Computer Engineering Portfolio Project

---

## 1. Abstract
This report surveys deep learning approaches for pneumonia classification from chest X-ray images. Pneumonia is a major global health risk, particularly for children and elderly cohorts. While X-ray imaging is fast and accessible, manual inspection is prone to subjective interpretation. Deep learning models, specifically Convolutional Neural Networks (CNNs), are highly suited for identifying dense consolidation areas in lung images. We review transfer learning using pretrained architectures (ResNet-50) and Explainable AI (Grad-CAM) to establish a baseline and state-of-the-art framework on the chest X-ray dataset.

## 2. Introduction
Teasing out pneumonia patterns (opacity, infiltration) in chest X-rays is a complex imaging task. Automated models can assist radiologists by screening images, prioritizing urgent cases, and providing a secondary verification step. This pre-research report explores baseline CNN configurations, pretrained models, and class activation mapping to design a reliable medical classification pipeline.

## 3. Problem Definition
The task is binary classification of chest X-ray images:
- **Input:** JPEG frontal view X-ray.
- **Output:** Binary prediction (`NORMAL` vs `PNEUMONIA`) and model confidence.
- **Clinical Integration:** Serving as a research-grade decision support prototype.

## 4. Literature Review and Benchmarks

| No | Study | Year | Dataset | Methodology | Strengths & Limitations |
| :--- | :--- | :---: | :--- | :--- | :--- |
| 1 | Kermany et al. | 2018 | Mendeley Chest X-Ray | CNN + Transfer Learning | Pediatric data; high performance. Limited multi-center verification. |
| 2 | Rajpurkar et al. | 2017 | ChestX-ray14 | DenseNet-121 | Radiologist level. Multi-label, lacks visual explanation. |
| 3 | Stephen et al. | 2019 | Kaggle Pneumonia | CNN from scratch + Augmentation | Simple CNN benchmark. No transfer learning. |
| 4 | Yadav & Jadhav | 2019 | Kaggle Pneumonia | SVM, VGG16, InceptionV3 | Detailed model comparison. Small data size. |
| 5 | Hashmi et al. | 2020 | Pediatric Chest X-Ray | Transfer Learning Comparison | Compares SqueezeNet and others. Sensitive to data split. |
| 6 | Selvaraju et al. | 2017 | ImageNet | Grad-CAM | Visualizes model activation. Not a clinical segmentation mask. |

## 5. Proposed Experimental Design

| Exp. | Model | Purpose | Expected Output |
| :--- | :--- | :--- | :--- |
| 1 | Simple CNN | Measure baseline performance | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 2 | ResNet-50 frozen | Test transfer learning effect | Validation accuracy and comparative metrics |
| 3 | ResNet-50 fine-tuned | Maximize model performance | Final classifier, error analysis, benchmark |
| 4 | Grad-CAM | Visual explanation | Class activation maps for correct and failed predictions |

## 6. Dataset Description
The "Chest X-Ray Images (Pneumonia)" dataset consists of 5,863 JPEG images split into train, val, and test. We will use a stratified random split (10% validation) to ensure classes are proportional in both training and validation sets, keeping the test set isolated.

## 7. References
1. Kermany, D. S. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5).
2. Rajpurkar, P. et al. (2017). CheXNet. *arXiv:1711.05225*.
3. Selvaraju, R. R. et al. (2017). Grad-CAM. *ICCV 2017*.