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

* **Asset Monitoring (Computer Vision):** The baseline ResNet18 model achieved 70.54% validation accuracy but suffered from overfitting. The advanced ResNet50 model, enhanced with aggressive augmentation, dropout, and two-phase fine-tuning, reached **74.33%** accuracy with stable, generalizable performance.
* **Customer Engagement (Recommender):** The baseline SVD model posted an RMSE of 1.4460 but recommended already-visited places. The professional tuned SVD model reduced RMSE to **1.4164** (a 1.83% improvement) and guarantees novel recommendations through corrected logic.

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

**Dataset Layout**
`<repo_root>/train/<class_name>/*.jpg`

### Methodology

* **Baseline (ResNet18):** Transfer learning with a frozen backbone and standard augmentation. Achieved 70.54% validation accuracy but exhibited severe overfitting (diverging validation loss).
* **Advanced (ResNet50):** ColorJitter, RandomAffine, Dropout (p=0.5), two-phase fine-tuning (unfreezing `layer3`/`layer4` with reduced learning rate), and early stopping. Achieved **74.33%** validation accuracy with converging losses.

### Training Performance

|                                 Baseline (ResNet18)                                 |                                      Advanced (ResNet50)                                      |
| :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| <img src="model_output_pytorch/training_performance_plots_pytorch.png" width="360"> | <img src="model_output_pytorch_advanced/training_performance_plots_advanced.png" width="360"> |

**Analysis**
Baseline curves show overfitting (validation loss rising while training loss falls). Advanced curves converge with a stable gap, indicating healthy generalization.

### Diagnostic Evaluation

<p align="center">
  <img src="model_output_pytorch_advanced/confusion_matrix_normalized.png" alt="Confusion Matrix (Normalized)" width="760">
</p>

The matrix emphasizes per-class behavior: a strong diagonal indicates reliable recognition; off-diagonals highlight confusions between similar classes (e.g., altar vs. apse). A full per-class precision/recall/F1 report is provided in `model_output_pytorch_advanced/classification_report.txt`. If a normalization warning appears, it indicates an absent class in the validation split; use stratified sampling in future iterations.

### Configuration

| Parameter      | Value                                |
| :------------- | :----------------------------------- |
| Framework      | PyTorch 2.5.x (+ CUDA 12.1)          |
| Input Size     | 150 × 150                            |
| Batch Size     | 32                                   |
| Optimizer      | Adam (reduced LR during fine-tuning) |
| Loss Function  | Cross-Entropy                        |
| Early Stopping | ~Epoch 7                             |

**Performance Summary**

| Metric              | Baseline (ResNet18) | Advanced (ResNet50) |
| :------------------ | :-----------------: | :-----------------: |
| Validation Accuracy |        70.54%       |      **74.33%**     |
| Overfitting         |        Severe       |      Mitigated      |

---

## 2. Professional Tourism Recommender

### Objective

Deliver accurate, novel travel recommendations by excluding previously rated destinations.

### Methodology

* **Baseline (Simple SVD):** Default parameters and basic split. RMSE **1.4460**; recommended already-visited places.
* **Professional (Tuned SVD):** GridSearchCV over epochs, learning rate, and regularization; strict novelty filter to exclude rated items. RMSE **1.4164** (1.83% improvement).

### Evaluation

<p align="center">
  <img src="plots_professional/recommender_precision_recall_at_k.png" alt="Precision and Recall@K" width="760">
</p>

Precision@K/Recall@K are constrained by **data sparsity** (~10k ratings across 300 users and 437 places). The priority for further gains is richer interaction data rather than incremental model tweaks.

**Performance Summary**

| Metric               |     Baseline     |     Professional     |
| :------------------- | :--------------: | :------------------: |
| RMSE (↓ better)      |      1.4460      |      **1.4164**      |
| Recommendation Logic | Repeats possible | **Novelty enforced** |
| User Value           |        Low       |         High         |

---

## 3. Environment & Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# Optional (GPU build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. How to Run

```bash
# Baseline classifier
python structure_classifier_pytorch.py

# Advanced classifier (ResNet50)
python structure_classifier_advanced.py

# Professional recommender
python tourism_recommender_professional.py
```

**Outputs**

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

Deterministic splits and fixed seeds:

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```

Minor variance may arise from GPU kernels. Artifacts (plots and reports) are programmatically generated.

---

## 7. Project Conclusion

* The advanced ResNet50 classifier overcame overfitting and achieved **74.33%** validation accuracy with robust generalization.
* The tuned SVD recommender reduced error to **1.4164 RMSE** and enforces novelty, improving user trust and value.
* Diagnostic findings (e.g., data sparsity) point to strategic data acquisition as the most impactful next step.

---

## 8. Roadmap

* Grad-CAM for interpretable classification.
* FastAPI microservice for the recommender.
* Mixed-precision training and cosine annealing.
* Expanded datasets across regions and structure types.
* CI/CD for automated validation.

---

## 9. License & Contact

**License:** MIT (see `LICENSE`)
**Author:** Daniel Allen, RN MBA MSHI
**Organization:** QMTRY LLC
**Contact:** [contracts@qmtry.com](mailto:contracts@qmtry.com)
**Website:** [https://www.qmtry.ai](https://www.qmtry.ai)
