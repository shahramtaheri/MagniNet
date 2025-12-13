MagniNet

Magnification-Specific Hybrid CNN–Transformer Framework for Breast Cancer Histopathology Classification

Overview

This repository provides the official PyTorch implementation of MagniNet, a magnification-specific hybrid CNN–Transformer framework for breast cancer histopathology image classification. The model is designed to jointly perform binary benign–malignant classification and eight-class histological subtype recognition, while maintaining strong interpretability and computational efficiency.

MagniNet integrates EfficientNet-based local feature extraction, cross-attention fusion, and a hierarchical Swin Transformer to capture both fine-grained cellular morphology and long-range contextual dependencies. Separate models are trained for each optical magnification (40×, 100×, 200×, and 400×), enabling scale-specific representation learning consistent with pathological diagnostic workflows.

The framework is evaluated on the BreakHis dataset and externally validated on BACH, Camelyon16, PatchCamelyon, and TCGA-BRCA, demonstrating robust generalization across staining protocols, acquisition devices, and magnification settings.

Key Contributions

Magnification-specific learning for multi-scale histopathology images

Hybrid CNN–Transformer architecture combining EfficientNet and Swin Transformer

Cross-attention fusion of local morphological and global contextual features

Dual-head classification for binary and multi-class tasks

Integrated interpretability using SHAP, Grad-CAM++, and LIME

External validation on multiple independent histopathology datasets

Lightweight architecture with low parameter count and fast inference

Repository Structure
MagniNet/
├── train_magninet.py
├── test_magninet.py
├── evaluate_external.py
├── infer.py
├── models/
├── datasets/
├── preprocessing/
├── interpretability/
├── analysis/
├── configs/
├── utils/
├── notebooks/
├── scripts/
├── results/
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── LICENSE


Each module is organized to clearly separate model definition, data handling, training, evaluation, interpretability, and analysis, ensuring full reproducibility.

Installation
Requirements

Python ≥ 3.9

PyTorch ≥ 2.0

timm

NumPy, SciPy, scikit-learn

matplotlib, seaborn

shap, lime

Install dependencies
pip install -r requirements.txt


or using Conda:

conda env create -f environment.yml
conda activate magninet

Datasets
Primary Dataset

BreakHis – Breast Cancer Histopathological Database
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

External Validation Datasets

BACH (ICIAR 2018)

Camelyon16

PatchCamelyon

TCGA-BRCA

All datasets are publicly available and must be downloaded separately in accordance with their respective licenses.

Data Organization

For each magnification, datasets should follow the structure:

data/BreakHis/40x/
├── train/
│   ├── images/
│   └── labels.json
├── val/
│   ├── images/
│   └── labels.json
└── test/
    ├── images/
    └── labels.json


Binary labels are either provided explicitly or derived from the eight-class annotations.

Training

MagniNet is trained independently for each magnification.

python train_magninet.py --config configs/magninet_40x.yaml


Training details:

Optimizer: AdamW

Learning rate: 1e-4

Scheduler: Cosine annealing

Mixed-precision training (AMP)

Loss:

𝐿
=
𝛼
𝐿
𝑚
𝑢
𝑙
𝑡
𝑖
+
(
1
−
𝛼
)
𝐿
𝑏
𝑖
𝑛
𝑎
𝑟
𝑦
,
𝛼
=
0.6
L=αL
multi
	​

+(1−α)L
binary
	​

,α=0.6
Evaluation
Internal Testing
python test_magninet.py --checkpoint best.pt

External Validation
python evaluate_external.py --checkpoint best.pt --dataset BACH


Evaluation metrics include accuracy, precision, recall, specificity, F1-score, and AUROC.

Interpretability

MagniNet integrates multiple interpretability techniques:

Grad-CAM++ for class-discriminative localization

SHAP for pixel-level attribution

LIME for superpixel-based explanations

Quantitative evaluation is performed using Pointing Game Accuracy and ROI Intersection-over-Union (IoU) against expert-annotated regions.

Computational Analysis

The repository includes scripts to compute:

Model parameters

FLOPs

Inference time

Statistical significance tests (McNemar’s test)

These analyses correspond to the efficiency and statistical validation reported in the manuscript.

Reproducibility

All experiments are fully reproducible using the provided configuration files, scripts, and documented training protocols. Random seeds are fixed and dataset splits are explicitly defined.

Citation

If you use this code, please cite:

Shahram Taheri.
MagniNet: A Magnification-Specific Hybrid CNN–Transformer Framework for Accurate Breast Cancer Histopathology Classification.
The Visual Computer, under review.


(Please update with the final DOI after publication.)

License

This repository is released for research and academic use. See the LICENSE file for details.

Contact

For questions or clarifications related to the implementation or experiments, please contact the corresponding author.
