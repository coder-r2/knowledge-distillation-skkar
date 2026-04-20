# Knowledge Distillation: Transferring Reasoning, Preserving Knowledge, and Enforcing Reliability

> **Official PyTorch implementation for the Knowledge Distillation research project by Team S.K.K.A.R.**
> Indian Institute of Science (IISc), Bengaluru.

Deploying reliable intelligence on resource-constrained edge devices demands models that are not only compact but structurally sound and trustworthy. This repository contains our comprehensive investigation into Knowledge Distillation (KD) across three distinct dimensions, reframing KD not merely as a compression tool, but as a principled framework for transferring spatial reasoning, preventing catastrophic forgetting, and enforcing predictive calibration.

## 🚀 The Three Hypotheses

### 1️⃣ Dark Knowledge Across Architectural Paradigms
We investigate whether soft-label "dark knowledge" encodes structural inter-class reasoning beyond simple accuracy gains, and if this reasoning can survive massive architectural gaps.
* **Homogeneous Distillation:** Demonstrated that a ResNet-18 student trained with KD surpasses its ResNet-50 teacher in Out-of-Distribution (OOD) robustness across CIFAR-10-C corruptions.
* **Heterogeneous Distillation (ViT -> CNN):** We broke the "translation firewall" of standard linear projectors using **Masked Generative Distillation (MGD)**. By masking 75% of the CNN's features, we forced a ResNet-18 to abandon its native local-texture bias and inherit the global shape geometry of a Vision Transformer (ViT-Small), achieving state-of-the-art Heavy Blur robustness (42.09%) on Imagenette.
* 🔗 *[See `Hypothesis 1` directory for full details, MGD architecture, and Grad-CAM visualizations]*

### 2️⃣ Incremental KD Using Feature Replay (Continual Learning)
We test the viability of distillation as the *sole* memory mechanism in continual learning, replacing raw data replay with lightweight feature constraints.
* **Setup:** A 5-step class-incremental protocol on MNIST (learning 2 new classes per step with no prior images).
* **Multi-Loss Strategy:** Combined new-class cross-entropy with replay cross-entropy, intermediate l2 distillation, logit-level KL divergence, and penultimate cosine distillation.
* **Results:** Proved that compact architectures regularize better. Our lightweight **CNN5** significantly outperformed the deeper ResNet-18 (84.97% vs 78.77% final accuracy), confirming that multi-teacher distillation effectively mitigates catastrophic forgetting without storing raw images.
* 🔗 *[See `Hypothesis 2` directory for sequential training pipelines and confusion matrices]*

### 3️⃣ Self-Distillation for Medical Edge Deployment
In high-stakes domains like medical imaging, overconfident neural networks are dangerous. We leverage Self-Distillation (SD) to train fast, accurate, and perfectly calibrated multi-exit models.
* **Architecture:** ResNet-50 (BloodMNIST) and ResNet-18 (Chest X-Ray) partitioned with intermediate bottleneck/classifier exits. The deepest layer acts as an internal teacher for the shallower blocks.
* **Inference Speed:** Shallow exits achieved massive inference latency reductions (e.g., a 63.5% reduction on ResNet-50 Exit 1) while maintaining or improving baseline accuracy.
* **Reliability & Calibration:** SD suppressed the dangerous overconfidence of standard cross-entropy. Expected Calibration Error (ECE) was drastically reduced (e.g., 3.52% -> 1.85% on BloodMNIST), pushing reliability curves closer to the ideal identity line.
* 🔗 *[See `Hypothesis 3` directory for training scripts, calibration curve generation, and CKA analysis]*

## 📂 Repository Structure
```text
├── Hypothesis 1/                    # Dark Knowledge & Heterogeneous Distillation
│   ├── Dark Knowledge Transfer in Heterogenous Distillation/
│   │   ├── train_vit-cnn_mgd.py     # Primary MGD ViT->CNN pipeline
│   │   ├── train_baselines.py       # Supervised & Projector baselines
│   │   └── experiments.ipynb        # Grad-CAM and OOD robustness evaluations
│   └── Dark Knowledge Transfer in Homogenous Distillation/
│
├── Hypothesis 2/                    # Continual Learning & Feature Replay
│   ├── cnn5_pipelines.py            # Main training pipeline for CNN5
│   ├── pipeline1resnet18.py         # ResNet-18 Single-Teacher pipeline
│   ├── pipeline2resent18.py         # ResNet-18 Multi-Teacher ensemble pipeline
│   └── hyperparametertuning.py      # Grid search across alpha, beta, gamma, delta
│
├── Hypothesis 3/                    # Self-Distillation & Calibration
│   ├── models.py                    # Multi-exit ResNet architectures
│   ├── losses.py                    # CE, KL Divergence, and L2 Hint Loss implementations
│   ├── train_sd.py                  # Self-Distillation training loop
│   ├── calibration_curve.py         # Reliability diagram plotting utilities
│   └── measure.py / entropy.py      # ECE, NLL, Brier Score, and predictive entropy metrics
│
├── Project_Report.pdf               # Full comprehensive paper
|── Project_Presentation.pdf         # Slides for project presentation
├── requirements.txt                 # Master dependencies file
└── README.md                        # Master repository overview
```

## 🛠️ Quick Start

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/coder-r2/knowledge-distillation-skkar.git
cd knowledge-distillation-skkar
pip install -r requirements.txt
```

**2. Navigating the Hypotheses:**
Each hypothesis functions as an independent module. Navigate to the respective folder to run its specific training or evaluation scripts. For example, to test the Self-Distillation models:
```bash
cd "Hypothesis 3"
python train_sd.py
```

## 👥 Team S.K.K.A.R.
* **Abinav Thangaraju Sethupathy**
* **Rishe Raghavendira G**
* **Kretik AS**
* **Khaja Aflal H**
* **CV Sai Charan**

*Indian Institute of Science (IISc), Bengaluru*

## 📚 Acknowledgements
We extend our deepest gratitude to our mentor, **Armaan Khetarpaul**, for his constant guidance throughout this project, and to **Dr. Aditya Gopalan** and **Dr. Shishir NYK** for providing us the opportunity to explore these exciting avenues in Knowledge Distillation.
