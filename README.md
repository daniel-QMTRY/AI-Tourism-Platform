
```markdown
# AI-Powered Tourism Platform
Deep Learning for Cultural Heritage Preservation & Intelligent Travel Recommendation  
**Historical Structures Image Classification (PyTorch) | Professional Tourism Recommender**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-3.8+-11557C" />
  <img src="https://img.shields.io/badge/scikit--surprise-1.1.4-FF9A00" />
  <img src="https://img.shields.io/badge/License-MIT-informational" />
</p>

---

## Executive Summary

This repository delivers two production-ready machine learning systems designed to transform the tourism industry. The first, a deep learning classifier, identifies historical structures from images, enabling efficient cultural heritage monitoring. The second, a collaborative-filtering recommender, curates personalized travel suggestions that spark discovery by ensuring every recommendation is a new adventure. Through a rigorous Build → Diagnose → Improve → Validate workflow, both systems evolved from promising prototypes into robust, defensible solutions.

- **Asset Monitoring (Computer Vision):** The baseline ResNet18 model achieved 70.54% validation accuracy but suffered from overfitting. The advanced ResNet50 model, enhanced with aggressive augmentation, dropout, and two-phase fine-tuning, reached **74.33%** accuracy with stable, generalizable performance.
- **Customer Engagement (Recommender):** The baseline SVD model posted an RMSE of 1.4460 but recommended already-visited places. The professional tuned SVD model reduced RMSE to **1.4164** (a 1.83% improvement) and guarantees novel recommendations through corrected logic.

These outcomes reflect a disciplined approach to problem-solving, delivering systems that balance technical precision with practical impact.

---

## Table of Contents

1. [Historical Structures Image Classification](#1-historical-structures-image-classification)
2. [Professional Tourism Recommender](#2-professional-tourism-recommender)
3. [Environment & Installation](#3-environment--installation)
4. [How to Run](#4-how-to-run)
5. [Repository Structure](#5-repository-structure)
6. [Reproducibility](#6-reproducibility)
7. [Project Conclusion](#7-project-conclusion)
8. [Roadmap](#8-roadmap)
9. [License & Contact](#9-license--contact)

---

## 1. Historical Structures Image Classification

### Objective
Classify images into eleven architectural categories (e.g., altar, bell tower, dome) to support automated cataloging and preservation of cultural heritage.

**Dataset Layout:**
```
<repo_root>/train/<class_name>/*.jpg
```

### Methodology
- **Baseline (ResNet18):** Employed transfer learning with a frozen backbone and standard augmentations. Achieved 70.54% validation accuracy but exhibited severe overfitting due to diverging validation loss.
- **Advanced (ResNet50):** Implemented ColorJitter, RandomAffine, Dropout (p=0.5), two-phase fine-tuning (unfreezing `layer3`/`layer4` with reduced learning rate), and early stopping. Achieved **74.33%** validation accuracy with converging losses, indicating strong generalization.

### Training Performance
| Baseline (ResNet18) | Advanced (ResNet50) |
|:---------------------|:--------------------|
| <img src="model_output_pytorch/training_performance_plots_pytorch.png" width="350"> | <img src="model_output_pytorch_advanced/training_performance_plots_advanced.png" width="350"> |

**Analysis:**
- **Baseline:** Diverging validation loss (orange) signals overfitting, as training loss drops excessively.
- **Advanced:** Converging training and validation losses with a stable gap confirm robust generalization.

### Diagnostic Evaluation
<p align="center">
  <img src="model_output_pytorch_advanced/confusion_matrix_normalized.png" alt="Confusion Matrix (Normalized)" width="700">
</p>

The confusion matrix highlights per-class performance, with a strong diagonal indicating reliable predictions. Off-diagonal elements reveal specific misclassifications for future refinement. The full `classification_report.txt` (in `model_output_pytorch_advanced/`) provides precision, recall, and F1-scores per class. Note: A normalization warning may indicate absent classes in the validation set, suggesting stratified sampling for future iterations.

### Configuration
| Parameter         | Value                          |
|:------------------|:-------------------------------|
| Framework         | PyTorch 2.5.x (+ CUDA 12.1)   |
| Input Size        | 150×150                       |
| Batch Size        | 32                            |
| Optimizer         | Adam (reduced LR for fine-tuning) |
| Loss Function     | Cross-Entropy                 |
| Early Stopping    | ~Epoch 7                      |

**Performance Summary:**
| Metric              | Baseline (ResNet18) | Advanced (ResNet50) |
|:--------------------|:---------------------|:--------------------|
| Validation Accuracy | 70.54%              | **74.33%**          |
| Overfitting         | Severe              | Mitigated           |

---

## 2. Professional Tourism Recommender

### Objective
Deliver accurate, novel travel recommendations by excluding previously rated destinations, ensuring a delightful user experience.

### Methodology
- **Baseline (Simple SVD):** Default parameters with a basic train-test split. Produced an RMSE of **1.4460** but recommended already-visited places, undermining user trust.
- **Professional (Tuned SVD):** Optimized via GridSearchCV (epochs, learning rate, regularization) and enforced novelty by filtering out rated items. Achieved **RMSE 1.4164** (1.83% improvement) with guaranteed fresh recommendations.

### Evaluation
<p align="center">
  <img src="plots_professional/recommender_precision_recall_at_k.png" alt="Precision and Recall@K" width="700">
</p>

**Analysis:** Precision@K and Recall@K scores reflect data sparsity (~10k ratings across 300 users and 437 places). While individual predictions are accurate, ranking true user preferences at the top is constrained by limited data. This insight prioritizes data acquisition over marginal model tuning for future improvements.

**Performance Summary:**
| Metric             | Baseline     | Professional  |
|:-------------------|:-------------|:-------------|
| RMSE (lower=better)| 1.4460       | **1.4164**   |
| Recommendation Logic| Repeats Possible | **Novelty Enforced** |
| User Value         | Low          | High         |

---

## 3. Environment & Installation

Set up the environment:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

**Optional (GPU Support):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Verify CUDA:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. How to Run

Execute the models:

```bash
# Baseline classifier
python structure_classifier_pytorch.py
# Advanced classifier
python structure_classifier_advanced.py
# Professional recommender
python tourism_recommender_professional.py
```

**Outputs:**
```
model_output_pytorch/
  └─ training_performance_plots_pytorch.png
model_output_pytorch_advanced/
  ├─ training_performance_plots_advanced.png
  ├─ confusion_matrix_normalized.png
  └─ classification_report.txt
plots_professional/
  ├─ age_distribution.png
  └─ recommender_precision_recall_at_k.png
```

---

## 5. Repository Structure

```
.
├── README.md
├── requirements.txt
├── structure_classifier_pytorch.py
├── structure_classifier_advanced.py
├── tourism_recommender.py
├── tourism_recommender_professional.py
├── model_output_pytorch/
├── model_output_pytorch_advanced/
└── plots_professional/
```

---

## 6. Reproducibility

Ensured through deterministic splits and fixed seeds:

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```

Minor variance may arise from GPU kernels. All artifacts (plots, reports) are programmatically generated for consistency.

---

## 7. Project Conclusion

This project showcases a disciplined machine learning lifecycle:
- The **Advanced ResNet50 classifier** overcame overfitting, achieving **74.33%** validation accuracy with robust generalization.
- The **Tuned SVD recommender** reduced RMSE to **1.4164** and ensured novel recommendations, enhancing user trust.
- Diagnostic insights (e.g., data sparsity) highlight the need for strategic data acquisition over incremental model tweaks.

This Build → Diagnose → Improve → Validate approach delivers production-grade AI systems ready for real-world impact.

---

## 8. Roadmap

- Implement Grad-CAM for interpretable classification visualizations.
- Deploy the recommender as a FastAPI service for real-time use.
- Explore mixed-precision training and cosine annealing for efficiency.
- Expand datasets to include diverse geographic and cultural contexts.
- Integrate CI/CD pipelines for automated validation.

---

## 9. License & Contact

**License:** MIT (see `LICENSE`)  
**Author:** Daniel Allen, RN MBA MSHI  
**Organization:** QMTRY LLC  
**Contact:** [contracts@qmtry.com](mailto:contracts@qmtry.com)  
**Website:** [https://www.qmtry.ai](https://www.qmtry.ai)

```



This README is now a polished, executive-grade document that balances technical rigor with accessibility, ready to impress on GitHub or in a professional setting. Let me know if you need further tweaks!
