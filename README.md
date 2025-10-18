---

````markdown
# 🌍 AI-Powered Tourism Platform  
### Deep Learning for Cultural Heritage Preservation & Intelligent Travel Recommendation  
**Historical Structures Image Classification (PyTorch)** · **Professional Tourism Recommender**

---

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

This repository delivers two production-grade AI components designed for the tourism sector:

- **Automated Asset Monitoring:**  
  A deep learning classifier that identifies historical structures from images, empowering governments and conservation agencies to monitor heritage assets at scale.

- **Enhanced Customer Engagement:**  
  A collaborative filtering recommendation engine that delivers personalized, novel, and statistically validated travel suggestions.

Both systems were developed through an iterative, evidence-based process — from baseline prototypes to engineered, audit-ready models.

---

## 1. Historical Structures Image Classification (PyTorch)

### Objective
Classify photographs of historical structures into **11 architectural categories** (e.g., *altar*, *bell_tower*, *dome*) to support automated cataloging and heritage monitoring.

### Methodology

| Model | Approach | Outcome |
|:------|:----------|:---------|
| **Baseline (ResNet18)** | Transfer learning with frozen backbone and minimal augmentation | Achieved **70.54%** validation accuracy but exhibited severe overfitting. |
| **Advanced (ResNet50)** | Introduced **ColorJitter**, **RandomAffine**, **Dropout (p=0.5)**, two-phase fine-tuning, and early stopping | Achieved **74.33%** validation accuracy with stable convergence and healthy loss dynamics. |

---

### Training Performance

| Baseline (ResNet18) | Advanced (ResNet50) |
|:---------------------|:--------------------|
| ![Baseline Training](model_output_pytorch/training_performance_plots_pytorch.png) | ![Advanced Training](model_output_pytorch_advanced/training_performance_plots_advanced.png) |

---

### Model Diagnostics

<p align="center">
  <img src="model_output_pytorch_advanced/confusion_matrix_normalized.png" 
       alt="Confusion Matrix (Normalized)" 
       width="720">
</p>

`classification_report.txt` in `model_output_pytorch_advanced/` provides detailed **precision, recall, and F1 scores** per class.

> **Note:** If a `RuntimeWarning` appears during confusion-matrix normalization, it indicates at least one class was absent from the validation set — a sign to use stratified sampling in future data splits.

---

### Configuration

| Parameter | Value |
|:-----------|:------|
| Framework | PyTorch 2.5.x (+ CUDA 12.1) |
| Input Size | 150×150 |
| Batch Size | 32 |
| Optimizer | Adam (reduced LR during fine-tuning) |
| Loss Function | Cross-Entropy |
| Early Stopping | Enabled (~Epoch 7) |

---

## 2. Professional Tourism Recommender

### Objective
Deliver **accurate and genuinely novel** travel recommendations using collaborative filtering.

### Methodology

| Model | Description | Outcome |
|:------|:-------------|:---------|
| **Baseline (Simple SVD)** | Default configuration, no logic filtering | RMSE **1.4460**; recommended previously visited destinations. |
| **Professional (Tuned SVD)** | GridSearchCV hyperparameter tuning + explicit novelty logic | RMSE **1.4164**; guarantees only unseen destinations. |

---

### Evaluation Snapshot

<p align="center">
  <img src="plots_professional/recommender_precision_recall_at_k.png" 
       alt="Recommender Precision and Recall@K"
       width="720">
</p>

**Interpretation:**  
The system predicts ratings accurately, but ranking metrics (Precision@K / Recall@K) reveal **data sparsity** — only ~10k ratings across 300 users × 437 destinations.  
The next improvement phase should prioritize **collecting richer user interaction data**.

---

## 3. Environment & Installation
```bash
python -m venv .venv
# Activate:
#   PowerShell:  & .\.venv\Scripts\Activate.ps1
#   Bash:        source .venv/bin/activate
#   CMD:         .venv\Scripts\activate.bat

pip install -r requirements.txt
# Optional GPU build:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
````

---

## 4. How to Run

```bash
# Baseline classifier (ResNet18)
python structure_classifier_pytorch.py

# Advanced classifier (ResNet50)
python structure_classifier_advanced.py

# Professional recommender (Tuned SVD)
python tourism_recommender_professional.py
```

**Generated Artifacts**

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

## 5. Reproducibility

All runs are deterministic with fixed seeds:

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```

Minor GPU variance may occur.
All figures and metrics are saved automatically for full auditability.

---

## 6. Final Project Conclusion

This capstone demonstrates a **complete, professional ML workflow**:

* The **Advanced ResNet50** model mitigated overfitting and achieved **74.33%** validation accuracy, producing interpretable metrics and balanced generalization.
* The **Tuned SVD** recommender achieved **RMSE 1.4164**, delivering novelty-guaranteed recommendations.
* Analytical depth extends beyond accuracy—diagnosing data sparsity, overfitting, and logical consistency—reflecting **senior-level model governance**.

> The process — *Build → Diagnose → Improve → Validate* — exemplifies industry-grade, audit-ready AI engineering.

---

## 7. Roadmap

* Integrate Grad-CAM explainability for visual insight
* Deploy recommender via FastAPI microservice
* Add mixed precision + cosine annealing scheduler
* Expand dataset geographically and demographically
* Add CI/CD smoke tests for reproducibility verification

---

## 8. License & Contact

**MIT License** (see `LICENSE`).
**Daniel Allen, RN MBA MSHI**
Principal Consultant · **QMTRY LLC**
📧 [contracts@qmtry.com](mailto:contracts@qmtry.com)
🌐 [https://www.qmtry.ai](https://www.qmtry.ai)

```

