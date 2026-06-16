# Experiment Results Summary

This summary compares the test set results of the Simple CNN baseline model and the ResNet-50 transfer learning model.

## Dataset

- **Dataset Used:** Chest X-Ray Images (Pneumonia)
- **Test Set Distribution:**

| Class | Number of Images |
|---|---:|
| NORMAL | 234 |
| PNEUMONIA | 390 |
| **Total** | **624** |

*Note: During training, the Kaggle `train` and the small `val` directories were pooled together, and a stratified validation split (10%) was performed. The Kaggle `test` split was held out independently for final performance evaluation.*

## Model Comparison

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Simple CNN | 0.7180 | 0.6904 | **0.9949** | 0.8151 | 0.9238 |
| **ResNet-50 + Fine-Tuning** | **0.9279** | **0.9021** | 0.9923 | **0.9451** | **0.9837** |

## Confusion Matrix Values

### Simple CNN:
| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** | 60 | 174 |
| **Actual PNEUMONIA** | 2 | 388 |

### ResNet-50 + Fine-Tuning:
| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** | 192 | 42 |
| **Actual PNEUMONIA** | 3 | 387 |

## Discussion & Key Findings

- **Simple CNN Baseline:** The baseline model achieves a very high Recall (**99.49%**), meaning it successfully detects almost all positive pneumonia cases. However, it suffers from a high False Positive rate (174 healthy patients misclassified as sick), leading to a low Precision (**0.6904**) and overall Accuracy (**71.80%**).
- **Fine-Tuned ResNet-50:** The transfer learning approach with fine-tuning delivers the best overall performance. The False Positives are dramatically reduced from 174 to 42 (significantly improving Precision to **0.9021**), while keeping the False Negatives extremely low at only 3. Accuracy increases to **92.79%**, and the ROC-AUC reaches **0.9837**, making it the primary model candidate for clinical decision support.

## Grad-CAM Visualizations for Error Analysis

Key Grad-CAM outputs saved for inspection:
- **True PNEUMONIA detection:** `outputs/gradcam_person100_bacteria_477.png`
- **True NORMAL detection:** `outputs/gradcam_IM-0001-0001.png`
- **Error analysis example:** `outputs/gradcam_IM-0022-0001.png`

*Conclusion: Grad-CAM heatmap overlays successfully confirm that the model focuses on the consolidation regions inside the lung cavities rather than arbitrary noise or bones. While highly valuable for interpreting model decisions and debugging, these maps should serve as decision support tools and not stand-alone diagnostic proof.*