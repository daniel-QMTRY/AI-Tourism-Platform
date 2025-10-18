```markdown
# AI-Powered Tourism Platform: Where History Meets Wanderlust
Historical Structures Image Classification (PyTorch) & Professional Tourism Recommender

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557C)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1.4-FF9A00)
![License: MIT](https://img.shields.io/badge/License-MIT-informational)

---

## Executive Summary
This repository delivers two production-oriented ML components for the tourism sector:

- **Asset Monitoring (Computer Vision):** Classifies images of historical structures to support scalable preservation workflows.  
- **Customer Engagement (Recommender):** Generates personalized, **novel** travel suggestions with audited statistical accuracy.

**Headline Results**
- **Classifier:** Baseline ResNet18 validated feasibility but overfit (**70.54%** validation). An advanced ResNet50 with stronger augmentation, dropout, two-phase fine-tuning, and early stopping reached **74.33%** with converging losses.  
- **Recommender:** Baseline SVD yielded **RMSE 1.4460** and suggested repeats. A tuned SVD achieved **RMSE 1.4164** and guarantees only new recommendations. Precision@K/Recall@K surfaces data sparsity as the main limiter for ranking quality.

---

## 1) Overview
Two problems, one disciplined workflow: **baseline → diagnose → improve → validate**.

- **Computer Vision:** Multi-class classification across eleven architectural categories (e.g., altar, bell_tower, dome).  
- **Recommendations:** Logically sound, statistically accurate suggestions that explicitly exclude previously rated items.

---

## 2) Historical Structures Classification (PyTorch)

### Problem
Classify images of historical structures into **11** categories to enable automated cataloging and monitoring.

**Expected dataset**
```

<repo_root>/train/<class_name>/*.jpg

````

### Methodology

**Baseline (ResNet18)**  
Frozen backbone + new linear head; standard augmentation.  
**Result:** **70.54%** validation accuracy with classic overfitting (validation loss diverges).

**Advanced (ResNet50)**  
ColorJitter + RandomAffine; Dropout (p=0.5); two-phase fine-tuning (head → unfreeze `layer3`/`layer4` at lower LR); early stopping.  
**Result:** **74.33%** validation accuracy; training/validation curves track closely.

### Training Curves

| Baseline (ResNet18)                                                                 | Advanced (ResNet50)                                                                      |
|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|
| ![Baseline](model_output_pytorch/training_performance_plots_pytorch.png)           | ![Advanced](model_output_pytorch_advanced/training_performance_plots_advanced.png)       |

### Post-Training Diagnostics

<p align="left">
  <img src="model_output_pytorch_advanced/confusion_matrix_normalized.png"
       alt="Confusion Matrix (Normalized)"
       width="760">
</p>

A per-class precision/recall/F1 report is saved at:
`model_output_pytorch_advanced/classification_report.txt`.

### Configuration
- PyTorch 2.5.x (+ cu121), 150×150 images, batch 32  
- Optimizer: Adam (head only; reduced LR for unfrozen layers)  
- Loss: Cross-Entropy, up to 50 epochs (early stop ≈ epoch 7)

**Key metrics**

| Metric              | Baseline (ResNet18) | Advanced (ResNet50) |
|---------------------|---------------------|---------------------|
| Validation Accuracy | 70.54%              | **74.33%**          |
| Overfitting         | Severe              | **Mitigated**       |

---

## 3) Professional Tourism Recommender

### Problem
Deliver **novel**, high-quality recommendations using collaborative filtering.

### Methodology

**Baseline (Simple SVD)**  
Default configuration; basic split.  
**Issues:** **RMSE 1.4460**; recommended previously rated items.

**Professional (Tuned SVD)**  
GridSearchCV over epochs/LR/regularization; strict novelty filter excluding all previously rated items.  
**Result:** **RMSE 1.4164** with strictly new recommendations.

### Ranking Quality

![Precision/Recall@K](plots_professional/recommender_precision_recall_at_k.png)

Interpretation: lower P@K/R@K scores primarily reflect **data sparsity** (∼10k ratings across ~300 users × 437 places). Improving ranking confidence depends more on collecting additional ratings than on marginal model tweaks.

---

## 4) Environment & Installation
```bash
python -m venv .venv
# PowerShell:  & .\.venv\Scripts\Activate.ps1
# cmd.exe:     "X:\Capstone_PRJCT\Capstone 2\.venv\Scripts\activate.bat"
# bash:        source .venv/bin/activate  (or .venv/Scripts/activate on Windows Git Bash)

pip install -r requirements.txt

# Optional GPU build (PyTorch wheels):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
````

---

## 5) How to Run

```bash
# Baseline classifier (ResNet18)
python structure_classifier_pytorch.py

# Advanced classifier (ResNet50 with dropout, fine-tuning, early stopping)
python structure_classifier_advanced.py

# Tourism recommender (tuned SVD)
python tourism_recommender_professional.py
```

**Outputs**

```
model_output_pytorch/
  training_performance_plots_pytorch.png

model_output_pytorch_advanced/
  training_performance_plots_advanced.png
  confusion_matrix_raw.png
  confusion_matrix_normalized.png
  classification_report.txt

plots_professional/
  age_distribution.png
  recommender_precision_recall_at_k.png
```

---

## 6) Repository Structure

```
.
├─ README.md
├─ requirements.txt
├─ structure_classifier_pytorch.py
├─ structure_classifier_advanced.py
├─ tourism_recommender_professional.py
├─ tourism_recommender.py
├─ model_output_pytorch/
├─ model_output_pytorch_advanced/
└─ plots_professional/
```

---

## 7) Reproducibility

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```

Minor variance may occur due to GPU kernels. All artifacts are saved for auditability.

---

## 8) Final Project Conclusion

The initiative delivered two robust, production-minded systems:

* **CV:** The advanced ResNet50 eliminated overfitting and improved validation accuracy to **74.33%**. Confusion-matrix and per-class metrics provide actionable diagnostics for data and labeling improvements.
* **RecSys:** The tuned SVD reduced error to **1.4164 RMSE** and enforces **novelty**. Precision@K/Recall@K identified **data sparsity** as the dominant limiter, guiding a data-collection roadmap as the highest-leverage next step.

This is a mature workflow—**Build → Diagnose → Improve → Critically Analyze**—turning prototypes into decision-ready systems.

---

## 9) Roadmap

Grad-CAM interpretability; FastAPI microservice; weight decay + cosine annealing + mixed precision; stratified splits and dataset expansion; CI smoke tests and artifact publishing.

---

## 10) License & Contact

**MIT License** (see `LICENSE`).
**Daniel Allen — QMTRY LLC**
[contracts@qmtry.com](mailto:contracts@qmtry.com) · [https://www.qmtry.ai](https://www.qmtry.ai)

```
```
