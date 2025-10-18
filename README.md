````markdown
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

This repository delivers two production-grade AI components designed for the tourism sector:

- **Automated Asset Monitoring**  
  A deep learning classifier that identifies historical structures from images, enabling cultural agencies to track, analyze, and preserve heritage assets.

- **Enhanced Customer Engagement**  
  A collaborative filtering recommendation system that produces personalized, novel, and statistically validated travel suggestions.

Both solutions follow a consistent engineering framework: establish a baseline, diagnose weaknesses, apply targeted improvements, and validate performance.

---

## 1. Historical Structures Image Classification (PyTorch)

### Objective
Develop a model capable of classifying images into eleven architectural categories (e.g., altar, bell_tower, dome) to enable automated cataloging and asset monitoring.

### Methodology

| Model | Description | Key Outcome |
|:------|:-------------|:-------------|
| **Baseline (ResNet18)** | Transfer learning with frozen backbone and standard augmentations | Achieved **70.54%** validation accuracy but exhibited severe overfitting. |
| **Advanced (ResNet50)** | ColorJitter, RandomAffine, Dropout (p=0.5), two-phase fine-tuning, and early stopping | Achieved **74.33%** validation accuracy with stable convergence and strong generalization. |

---

### Training Performance

| Baseline (ResNet18) | Advanced (ResNet50) |
|:---------------------|:--------------------|
| ![Baseline Training](model_output_pytorch/training_performance_plots_pytorch.png) | ![Advanced Training](model_output_pytorch_advanced/training_performance_plots_advanced.png) |

---

### Diagnostic Evaluation

<p align="center">
  <img src="model_output_pytorch_advanced/confusion_matrix_normalized.png"
       alt="Confusion Matrix (Normalized)"
       width="700">
</p>

The confusion matrix and classification report provide per-class precision, recall, and F1-score, enabling data-driven model refinement.  
If normalization produces a runtime warning, it indicates a class absent in the validation set, suggesting the use of stratified sampling in future splits.

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

**Performance Summary**

| Metric | Baseline (ResNet18) | Advanced (ResNet50) |
|:--------|:---------------------|:--------------------|
| Validation Accuracy | 70.54% | 74.33% |
| Overfitting | Severe | Mitigated |

---

## 2. Professional Tourism Recommender

### Objective
Generate statistically accurate and logically sound travel recommendations, ensuring every suggestion introduces a new, unvisited destination.

### Methodology

| Model | Description | Outcome |
|:------|:-------------|:--------|
| **Baseline (Simple SVD)** | Default configuration with basic split | RMSE **1.4460**, recommended previously rated destinations. |
| **Professional (Tuned SVD)** | GridSearchCV optimization; explicit filtering of previously rated items | RMSE **1.4164**, strictly novel recommendations. |

---

### Evaluation

<p align="center">
  <img src="plots_professional/recommender_precision_recall_at_k.png"
       alt="Recommender Precision and Recall@K"
       width="700">
</p>

Lower Precision@K and Recall@K scores are attributed to **data sparsity** (~10k ratings, 300 users × 437 places).  
The model demonstrates accurate individual predictions but limited ranking confidence, emphasizing the strategic need for additional user data.

---

## 3. Environment & Execution

**Dependencies**
```bash
python -m venv .venv
# Activate environment
# Windows PowerShell:  & .\.venv\Scripts\Activate.ps1
# Linux/Mac:           source .venv/bin/activate
pip install -r requirements.txt
````

**Run Commands**

```bash
python structure_classifier_pytorch.py          # Baseline classifier
python structure_classifier_advanced.py         # Advanced ResNet50 classifier
python tourism_recommender_professional.py      # Tuned recommender
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
  └─ recommender_precision_recall_at_k.png
```

---

## 4. Reproducibility

Experiments are deterministic and reproducible:

```python
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```

Minor variance may occur from GPU kernels; all artifacts are versioned for audit consistency.

---

## 5. Final Project Conclusion

This project demonstrates a complete, audit-ready machine learning lifecycle.

* The **Advanced ResNet50** classifier effectively overcame overfitting and achieved **74.33%** validation accuracy.
* The **Tuned SVD** recommender reduced RMSE to **1.4164**, providing verified novel recommendations.
* Evaluation revealed that future gains depend on **data enrichment**, not incremental tuning—reflecting strategic insight at a professional level.

This work exemplifies the discipline of **Build → Diagnose → Improve → Validate**, transforming prototypes into production-grade, defensible AI systems.

---

## 6. Roadmap

* Integrate Grad-CAM for visual interpretability
* Deploy recommender via FastAPI service
* Add mixed-precision training and cosine annealing scheduling
* Expand datasets across geographic and cultural domains
* Implement CI/CD testing for continuous validation

---

## 7. License & Contact

**License:** MIT License (see `LICENSE`)
**Author:** Daniel Allen, RN MBA MSHI
**Organization:** QMTRY LLC
**Contact:** [contracts@qmtry.com](mailto:contracts@qmtry.com)
**Website:** [https://www.qmtry.ai](https://www.qmtry.ai)

```

