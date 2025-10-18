AI-Powered Tourism Platform: Where History Meets Wanderlust


Executive Summary: Your Ticket to Smarter Tourism
Welcome to the future of travel, where cutting-edge AI transforms how we preserve history and plan adventures. This repository is your backstage pass to two powerhouse ML components:

Preserving the Past: A deep learning model that classifies images of historical structures with surgical precision, making asset monitoring a breeze.
Inspiring the Future: A slick recommender system that serves up personalized travel suggestions so fresh, they’ll spark wanderlust in even the most seasoned globetrotter.

Key Wins:

Image Classifier: Our baseline ResNet18 laid the groundwork but stumbled with overfitting (70.54% val. accuracy). Enter ResNet50, armed with aggressive augmentation and fine-tuning wizardry, hitting a robust 75.50% accuracy with losses that play nice together.
Recommender: The baseline SVD was a bit of a buzzkill (RMSE 1.4460, recommending déjà vu). Our tuned SVD flips the script with RMSE 1.4195 and only novel destinations.


Table of Contents

Overview
Part 1: Historical Structures Classification
Part 2: Professional Tourism Recommender
Environment & Installation
How to Run
Repository Structure
Reproducibility Checklist
Roadmap
License & Contact


Overview
Think of this repo as your AI travel buddy: it’s got a sharp eye for history and a knack for suggesting your next adventure. We tackled two challenges with a streamlined workflow: baseline → diagnose → upgrade → validate. 

Computer Vision: Classifies images into 11 architectural categories (think altars, domes, and bell towers) to automate cataloging and preservation.
Recommendations: Delivers statistically tight, logically sound travel suggestions that won’t bore you with places you’ve already checked off.


Part 1: Historical Structures Classification (PyTorch)
The Mission
Turn images of historical structures into tidy categories for scalable monitoring. From altars to vaults, we’re sorting 11 architectural classes with style.
Dataset Layout:
<repo_root>/train/<class_name>/*.jpg

The Journey
Baseline (ResNet18): We started with a frozen backbone and a fresh classifier head, spiced up with standard augmentation. It hit 70.54% validation accuracy but threw a tantrum with diverging losses—classic overfitting drama.
Advanced (ResNet50): We brought in the big guns—ColorJitter, RandomAffine, Dropout (p=0.5), two-phase fine-tuning (unfreezing layer3/layer4 with a lower learning rate), and early stopping. The result? A cool 75.50% accuracy, with training and validation losses dancing in sync.
Performance Visuals



Baseline (ResNet18)
Advanced (ResNet50)







What’s the Story?

Baseline: Training loss plummets, validation loss sulks → overfitting alert.
Advanced: Losses converge like old friends, with a stable gap → generalization for the win.

Tech Specs

Framework: PyTorch 2.5.x (+ CUDA 12.1)
Input: 150×150 images, batch size 32
Optimizer: Adam (head-only initially; fine-tuning with reduced LR)
Loss: Cross-Entropy, max 50 epochs (early stopping ~epoch 7)

Metrics Snapshot:



Metric
Baseline (ResNet18)
Advanced (ResNet50)
Δ



Validation Accuracy
70.54%
75.50%
+4.96 pts


Overfitting
Severe
Mitigated
—



Part 2: Professional Tourism Recommender
The Mission
Craft travel recommendations that feel like a personal concierge, not a broken record. We’re talking novel destinations with pinpoint accuracy.
The Journey
Baseline (Simple SVD): Default settings, basic split. It clocked an RMSE of 1.4460 but kept suggesting places users already visited—yawn.
Professional (Tuned SVD): We cranked up the charm with GridSearchCV over epochs, learning rate, and regularization, plus logic to banish previously rated spots. The result? RMSE 1.4195 (a 1.83% improvement) and recommendations that feel like a fresh adventure.
Performance Snapshot



Feature
Baseline
Professional
Impact



RMSE (error)
1.4460
1.4195
More precise


Rec. Logic
Flawed
Sound
Guarantees novelty


User Value
Low
High
Trustworthy discovery


Exploratory Data Analysis
Check out this age distribution plot to get a vibe for our user base:

Environment & Installation
Ready to dive in? Set up your environment faster than you can pack a suitcase:
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# Optional GPU boost:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Verify CUDA:
# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"


How to Run
Launch your AI travel assistant with these commands:
# Baseline classifier
python structure_classifier_pytorch.py
# Advanced classifier (ResNet50 with all the bells and whistles)
python structure_classifier_advanced.py
# Tourism recommender
python tourism_recommender_professional.py

Outputs:
model_output_pytorch/*.png
model_output_pytorch_advanced/*.png
plots_professional/*.png


Repository Structure
.
├── README.md
├── requirements.txt
├── structure_classifier_pytorch.py
├── structure_classifier_advanced.py
├── tourism_recommender_professional.py
├── tourism_recommender.py
├── model_output_pytorch/
├── model_output_pytorch_advanced/
└── plots_professional/


Reproducibility Checklist
We’ve got your back for consistent results:

Deterministic Splits: Using random_split with seeds:

import torch, random, numpy as np
torch.manual_seed(42); random.seed(42); np.random.seed(42)


Note: Minor GPU kernel variance may occur.
Artifacts: Plots and weights (if saved) are programmatically generated.


Roadmap
What’s next? Think of this as our travel itinerary:

Add Grad-CAM for visual explainability (because who doesn’t love a good heatmap?).
Deploy the recommender as a FastAPI service for real-time wanderlust.
Experiment with weight decay, cosine annealing, and AMP for extra pizzazz.
Expand the dataset for even sharper models.
Set up CI smoke tests to keep things smooth.


License & Contact
License: MIT (see LICENSE for details).Contact: Daniel Allen — QMTRY LLC — contracts@qmtry.com — https://www.qmtry.ai
Let’s make history and travel unforgettable, one model at a time!
