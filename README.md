# AI-Powered Tourism Platform
Historical Structures Image Classification (PyTorch) & Professional Tourism Recommender

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557C)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1.4-FF9A00)
![License: MIT](https://img.shields.io/badge/License-MIT-informational)

## Executive Summary
Two production-grade ML components for tourism:

- **Automated Asset Monitoring:** A CNN classifier recognizes historical structures from images, supporting cataloging and preservation.
- **Enhanced Customer Engagement:** A collaborative-filtering recommender delivers personalized, *novel* suggestions with audited accuracy.

**Results**
- **Classifier:** Baseline ResNet18 reached **70.54%** validation accuracy but overfit. The advanced ResNet50 (aggressive augmentation, dropout, two-phase fine-tuning, early stopping) achieved **75.50%** with converging loss curves.
- **Recommender:** Baseline SVD **RMSE 1.4460** and repeated items; tuned SVD with logic fix achieved **RMSE 1.4195** and guarantees only *new* recommendations.

## Table of Contents
1. Overview  
2. Part 1: Historical Structures Classification  
3. Part 2: Professional Tourism Recommender  
4. Environment & Installation  
5. How to Run  
6. Repository Structure  
7. Reproducibility Checklist  
8. Roadmap  
9. License & Contact

## 1) Overview
- **Computer Vision:** Multi-class classification of architectural images into eleven categories.  
- **Recommender Systems:** Logically sound, statistically accurate recommendations that exclude previously rated items.  
Lifecycle: **baseline → diagnosis → targeted improvements → validation**.

## 2) Part 1 — Historical Structures Classification (PyTorch)
**Challenge.** Classify images into **11** categories (e.g., altar, bell_tower, dome).  
**Dataset layout**
<repo_root>/train/<class_name>/*.jpg

markdown
Copy code

**Baseline (ResNet18).** Frozen backbone + new head; standard augmentation.  
Result: **70.54%** validation accuracy; validation loss diverges → overfitting.

**Advanced (ResNet50).** ColorJitter + RandomAffine; Dropout p=0.5; two-phase fine-tuning (unfreeze `layer3`/`layer4` at lower LR); early stopping.  
Result: **75.50%**; train/val losses decrease together; gap stable.

**Config.** PyTorch 2.5.x (+cu121), 150×150 images, batch 32, Adam, CE loss, up to 50 epochs (early stop ~epoch 7).

**Figures** (commit these paths):



markdown
Copy code

## 3) Part 2 — Professional Tourism Recommender
**Baseline SVD.** Default; **RMSE 1.4460**; recommended already-rated places.  
**Tuned SVD.** GridSearchCV over epochs/LR/regularization; logic excludes rated items.  
Result: **RMSE 1.4195** (−1.83%); output is strictly novel.

**Comparative**
| Feature | Baseline | Professional | Impact |
|---|---:|---:|---|
| RMSE | 1.4460 | **1.4195** | More precise |
| Logic | Flawed | **Sound** | Guarantees novelty |
| UX | Confusing | **Valuable** | Trustworthy |

## 4) Environment & Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Optional GPU build:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
5) How to Run
bash
Copy code
# Baseline classifier
python structure_classifier_pytorch.py
# Advanced classifier
python structure_classifier_advanced.py
# Tourism recommender
python tourism_recommender_professional.py
Outputs:

markdown
Copy code
model_output_pytorch/*.png
model_output_pytorch_advanced/*.png
plots_professional/*.png
6) Repository Structure
bash
Copy code
.
├─ README.md
├─ requirements.txt
├─ structure_classifier_pytorch.py
├─ structure_classifier_advanced.py
├─ tourism_recommender_professional.py
├─ tourism_recommender.py
├─ model_output_pytorch/                 # plots (png)
├─ model_output_pytorch_advanced/        # plots (png)
└─ plots_professional/                   # EDA (png)
7) Reproducibility Checklist
Fixed seeds recommended:

python
Copy code
import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)
Deterministic split via random_split. Minor variance may occur due to GPU kernels. Artifacts are saved programmatically.

8) Roadmap
Grad-CAM interpretability; FastAPI service for the recommender; weight decay + cosine annealing + AMP; dataset expansion; CI smoke tests.

9) License & Contact
MIT License (see LICENSE).
Daniel Allen — QMTRY LLC — contracts@qmtry.com — https://www.qmtry.ai

yaml
Copy code

---

## Add `.gitattributes` (silence line-ending churn)

Create `.gitattributes`:

```gitattributes
* text=auto
*.py text eol=lf
*.md text eol=lf
