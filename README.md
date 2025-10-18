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

- **Asset Monitoring (CV):** a deep learning model that classifies historical structures from images, enabling scalable preservation workflows.
- **Customer Engagement (RecSys):** a collaborative-filtering engine that produces personalized, **novel** travel suggestions with audited accuracy.

**Headline Results**
- **Classifier:** Baseline ResNet18 overfit (**70.54%** val). Advanced ResNet50 with aggressive augmentation, dropout, two-phase fine-tuning, and early stopping reached **74.33%** with converging losses.
- **Recommender:** Baseline SVD had **RMSE 1.4460** and suggested repeats. Tuned SVD achieved **RMSE 1.4164** and recommends **only new** places. Precision@K/Recall@K analysis revealed **data sparsity** as the ranking limiter—informing a data-collection roadmap.

---

## Table of Contents
1. Overview  
2. Part 1 — Historical Structures Classification  
3. Part 2 — Professional Tourism Recommender  
4. Environment & Installation  
5. How to Run  
6. Repository Structure  
7. Reproducibility Checklist  
8. Final Project Conclusion  
9. Roadmap  
10. License & Contact

---

## 1) Overview
Two problems, one disciplined workflow: **baseline → diagnose → improve → validate**.

- **Computer Vision:** multi-class classification into eleven architectural categories (e.g., altar, bell_tower, dome).  
- **Recommendations:** logically sound, statistically accurate suggestions that explicitly exclude previously rated items.

---

## 2) Part 1 — Historical Structures Classification (PyTorch)

### Challenge
Classify images of historical structures into **11** categories to support automated cataloging and monitoring.

**Dataset layout**
```

<repo_root>/train/<class_name>/*.jpg

````

### Methodology

**Baseline (ResNet18)**  
Frozen backbone + new classifier head; standard augmentation.  
**Result:** **70.54%** validation accuracy; validation loss diverges while training loss falls → **overfitting**.

**Advanced (ResNet50)**  
ColorJitter + RandomAffine; Dropout (p=0.5).  
Two-phase fine-tuning (train head → unfreeze `layer3`/`layer4` at lower LR).  
Early stopping to capture the best epoch.  
**Result:** **74.33%** validation accuracy; train/val losses track closely.

### Performance Figures

| Baseline (ResNet18)                                                                 | Advanced (ResNet50)                                                                      |
|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|
| ![Baseline: Training vs Validation](model_output_pytorch/training_performance_plots_pytorch.png) | ![Advanced: Training vs Validation](model_output_pytorch_advanced/training_performance_plots_advanced.png) |

**Advanced Evaluation (post-training)**

- Normalized confusion matrix and full per-class report:

<p align="left">
  <img src="model_output_pytorch_advanced/confusion_matrix_normalized.png"
       alt="Confusion Matrix (Normalized)"
       width="720">
</p>

The text report `model_output_pytorch_advanced/classification_report.txt` provides per-class precision/recall/F1 for targeted remediation.

> **Note on warnings:** If a `RuntimeWarning` appears during normalization, at least one class has zero samples in the validation split. Use **stratified sampling** in future splits.

### Configuration
- PyTorch 2.5.x (+ cu121), 150×150 images, batch 32  
- Optimizer: Adam (head only; reduced LR when unfreezing)  
- Loss: Cross-Entropy, up to 50 epochs (**early stop ≈ epoch 7**)

**Key metrics**

| Metric              | Baseline (ResNet18) | Advanced (ResNet50) |
|---------------------|---------------------|---------------------|
| Validation Accuracy | 70.54%              | **74.33%**          |
| Overfitting         | Severe              | **Mitigated**       |

---

## 3) Part 2 — Professional Tourism Recommender

### Challenge
Deliver **novel**, high-quality recommendations using collaborative filtering.

### Methodology

**Baseline (Simple SVD)**  
Default configuration on a basic split.  
**Issues:** **RMSE 1.4460**; recommended already-rated items.

**Professional (Tuned SVD)**  
GridSearchCV over epochs/LR/regularization; strict logic to exclude all previously rated items.  
**Result:** **RMSE 1.4164**; output is **strictly new** to the user.

### Evaluation Figures

- **Precision@K / Recall@K** (holdout evaluation; relevant ≥ 4 stars):

![Recommender Precision/Recall@K](plots_professional/recommender_precision_recall_at_k.png)

**Interpretation:** Low P@K/R@K reflects **data sparsity** (∼10k ratings; ~300 users × 437 places). With a sparse user-item matrix, ranking confidence suffers even when single-rating RMSE is strong—guiding investment toward **collecting more ratings**, not micro-tuning hyperparameters.

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

# Verify CUDA
# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
````

---

## 5) How to Run

```bash
# Baseline classifier (ResNet18)
python structure_classifier_pytorch.py

# Advanced classifier (ResNet50: dropout, fine-tuning, early stopping)
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

## 7) Reproducibility Checklist

* Deterministic validation split via `random_split`; prefer **stratified** splits for balanced class presence.
* Recommended seeds:

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```

* Minor variance may occur due to GPU kernels; artifacts are saved programmatically.

---

## 8) Final Project Conclusion

This initiative delivered two robust systems:

* **Part 1 (CV):** The advanced ResNet50 eliminated overfitting and improved validation accuracy to **74.33%**. Confusion-matrix and per-class metrics provide granular diagnostics to guide dataset and labeling improvements.
* **Part 2 (RecSys):** The tuned SVD lowered error to **1.4164 RMSE** and enforces **novelty**. Precision@K/Recall@K uncovered **data sparsity** as the dominant limiter—pointing to a data-collection strategy as the high-leverage next step.

This is a mature workflow: **Build → Diagnose → Improve → Critically Analyze**, converting prototypes into defensible, decision-ready systems.

---

## 9) Roadmap

Grad-CAM interpretability; FastAPI microservice for the recommender; weight decay + cosine annealing + mixed precision; stratified splits and dataset expansion; CI smoke tests and artifact publishing.

---

## 10) License & Contact

MIT License (see `LICENSE`).
Daniel Allen — QMTRY LLC — [contracts@qmtry.com](mailto:contracts@qmtry.com) — [https://www.qmtry.ai](https://www.qmtry.ai)

````

