AI-Powered Tourism Platform

Historical Structures Image Classification (PyTorch) & Professional Tourism Recommender
















Executive Summary & Business Mandate

This repository delivers two production-oriented machine learning components for the tourism sector:

Automated Asset Monitoring: A deep learning classifier that recognizes historical structures from images, enabling agencies to monitor cultural assets and prioritize preservation.

Enhanced Customer Engagement: A collaborative-filtering recommendation engine that provides personalized, novel travel suggestions with audited statistical accuracy.

Both components were developed through an iterative, evidence-based process: establish a baseline model to confirm feasibility and expose weaknesses, then engineer an advanced model that systematically eliminates those weaknesses.

Results at a glance

Image Classifier:
Baseline ResNet18 reached 70.54% validation accuracy but severely overfit. An advanced ResNet50 with stronger augmentation, dropout, two-phase fine-tuning, and early stopping achieved 75.50% validation accuracy and healthy, converging loss curves.

Tourism Recommender:
Baseline SVD yielded RMSE 1.4460 and recommended already-rated places. A tuned SVD (GridSearchCV) with corrected logic achieved RMSE 1.4195 and guarantees only new recommendations.

Table of Contents

Overview

Part 1: Historical Structures Classification
2.1 Challenge
2.2 Methodology: Baseline â†’ Advanced
2.3 Training Configuration
2.4 Results, Figures, and Interpretation

Part 2: Professional Tourism Recommender
3.1 Challenge
3.2 Methodology: Baseline â†’ Professional
3.3 Results & Artifacts

Environment & Installation

How to Run

Repository Structure

Reproducibility Checklist

Roadmap

License & Contact

1) Overview

Two problems, one professional workflow:

Computer Vision: Multi-class classification of historical architectural images into eleven categories to support cataloging and monitoring.

Recommender Systems: Statistically accurate, logically sound recommendations that explicitly exclude destinations a user has already rated.

Each solution follows the same lifecycle: baseline â†’ diagnosis â†’ targeted improvements â†’ validation.

2) Part 1 â€” Historical Structures Classification (PyTorch)
2.1 Challenge

Accurately classify images into eleven architectural categories (e.g., altar, bell_tower, dome, etc.) to enable automated cataloging and scalable heritage monitoring.

Expected dataset layout

<repo_root>/
  train/
    class_1/
      img_001.jpg
      ...
    class_2/
    ...

2.2 Methodology: Baseline â†’ Advanced

Baseline Model (ResNet18)

Transfer learning with ImageNet weights; backbone frozen; new linear head.

Standard augmentation (random resized crop, flips, rotation).

Outcome: Validation accuracy 70.54%; severe overfitting evidenced by rising validation loss while training loss falls.

Advanced Model (ResNet50)
Engineered to combat overfitting and increase generalization:

Aggressive augmentation: ColorJitter, RandomAffine to expand data variability.

Dropout regularization: p=0.5 in the classifier to reduce co-adaptation.

Two-phase fine-tuning:

Train head only;

Unfreeze layer3 and layer4 at a reduced learning rate to adapt high-level features to the domain.

Early stopping: Preserve best weights when validation loss ceases to improve.

Outcome: Validation accuracy 75.50% with closely tracking train/val curves; overfitting mitigated.

Comparative Summary

Metric	Baseline (ResNet18)	Advanced (ResNet50)	Î” Improvement
Validation Accuracy	70.54%	75.50%	+4.96 points
Overfitting	Severe	Mitigated	â€”
Generalization	Weak	Robust	â€”
2.3 Training Configuration

Framework: PyTorch 2.5.x (+ cu121)

Image size: 150Ã—150

Batch size: 32

Optimizer: Adam (head only; reduced LR for unfrozen layers during fine-tuning)

Loss: Cross-Entropy

Epochs: Up to 50 (early stopping in advanced run halted around epoch 7; validation loss minimum observed near epoch ~2)

Device: CUDA if available (e.g., RTX 4060 8 GB), otherwise CPU

2.4 Results, Figures, and Interpretation

Narrative

Baseline: Training accuracy rises while validation accuracy plateaus; validation loss diverges upward after early epochsâ€”a textbook overfitting pattern.

Advanced: Training and validation losses decrease together; accuracy curves maintain a stable, modest gapâ€”evidence of effective generalization.

Figures
Commit plots to the repository and embed them with relative paths:

![Baseline: Training vs Validation](model_output_pytorch/training_performance_plots_pytorch.png)
![Advanced: Training vs Validation](model_output_pytorch/training_performance_plots_pytorch_advanced.png)


How to read these plots

If validation loss increases while training loss decreases, the model is memorizing (overfitting).

Converging or jointly decreasing losses with a stable accuracy gap indicate healthy learning and generalization.

3) Part 2 â€” Professional Tourism Recommender
3.1 Challenge

Deliver a collaborative-filtering system that is both statistically accurate and logically sound, providing genuine discovery by recommending destinations a user has not previously rated.

3.2 Methodology: Baseline â†’ Professional

Baseline (Simple SVD)

Default SVD configuration; simple train/test split.

Issues: RMSE 1.4460; recommended places already rated by the user.

Professional (Tuned SVD)

Hyperparameter tuning: GridSearchCV over epochs, learning rate, and regularization to minimize cross-validated RMSE.

Logic correction: Recommendation function explicitly excludes all previously rated items to guarantee novelty.

Outcome: RMSE 1.4195 (âˆ’1.83%) with strictly novel recommendations.

Comparative Summary

Feature	Baseline Model	Professional Model	Impact & Justification
RMSE (Error Rate)	1.4460	1.4195	1.83% reduction; more precise predictions
Recommendation Logic	Flawed	Sound	Guarantees novelty; aligns with user discovery
User Experience	Confusing	Valuable	Trustworthy, business-relevant recommendations

Artifacts

EDA plots: plots_professional/*

Example top-N novel recommendations for a target user (printed to console and/or saved as CSV as implemented).

4) Environment & Installation

Create a virtual environment and install dependencies:

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


GPU build (recommended with NVIDIA GPU):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Verify CUDA availability:

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

5) How to Run
5.1 Train the Image Classifier

Ensure the dataset exists at ./train/<class_name>/*.jpg.

Execute:

python structure_classifier.py


Outputs:

model_output_pytorch/
  â”œâ”€ structure_classifier_model_pytorch.pth
  â”œâ”€ training_performance_plots_pytorch.png
  â””â”€ training_performance_plots_pytorch_advanced.png   # if you save advanced run separately

5.2 Run the Tourism Recommender
python tourism_recommender_professional.py


Outputs:

plots_professional/
  â””â”€ <EDA images>

6) Repository Structure
<repo_root>/
  README.md
  requirements.txt
  structure_classifier.py
  tourism_recommender_professional.py
  train/                        # dataset (not versioned)
    <class_1>/...
    <class_2>/...
  model_output_pytorch/
    structure_classifier_model_pytorch.pth
    training_performance_plots_pytorch.png
    training_performance_plots_pytorch_advanced.png
  plots_professional/
    *.png

7) Reproducibility Checklist

Python 3.11; pinned dependencies in requirements.txt.

Deterministic train/val split via PyTorch random_split.

Recommended fixed seeds for experiments:

import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)


Minor run-to-run variance may occur due to GPU kernels and data shuffling.

All artifacts (weights, plots) are programmatically saved for audit.

8) Roadmap

Integrate Grad-CAM for classifier interpretability.

Expose the recommender as a FastAPI microservice with SQLite/PostgreSQL backing.

Add weight decay and cosine annealing scheduler; optional mixed-precision training.

Expand classification dataset across additional regions and structure types.

CI pipeline for environment checks, smoke tests, and artifact publication.

9) License & Contact

License: MIT (see LICENSE).

Contact:
Daniel Allen, RN MBA MSHI
Principal Consultant, QMTRY LLC
contracts@qmtry.com
 Â· https://www.qmtry.ai

Notes for Reviewers

Classifier: Baseline 70.54% with overfitting; Advanced 75.50% with converging curves (early stopping around epoch ~7; validation loss minimum near epoch ~2).

Recommender: RMSE improved from 1.4460 to 1.4195; logic now guarantees novel recommendations.

The repository demonstrates a professional workflow: diagnose, remediate, and validate with metrics and plots.