# Real-time Domain Adaptation in Semantic Segmentation

This repository contains the implementation and code for the paper **"Real-time Domain Adaptation in Semantic Segmentation"** by Luca Ianniello, Raffaele Martone, and Antonio Sirica (January 30, 2025). The project investigates domain adaptation techniques for semantic segmentation using state-of-the-art models like DeepLabV2 and PIDNet on the LoveDA dataset. It explores methods such as data augmentation, adversarial learning, DACS, PEM, and CycleGAN to tackle domain shifts between urban and rural environments.

---

## Table of Contents

- [Real-time Domain Adaptation in Semantic Segmentation](#real-time-domain-adaptation-in-semantic-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Requirements](#requirements)
  - [Usage Instructions](#usage-instructions)
    - [Running Models](#running-models)
  - [Experiments and Results](#experiments-and-results)
    - [DeepLabV2 Experiments](#deeplabv2-experiments)
    - [PIDNet Experiments](#pidnet-experiments)
    - [Data Augmentation Experiments](#data-augmentation-experiments)
    - [Adversarial Learning Experiments](#adversarial-learning-experiments)
  - [| PIDNet Best Aug| 0.3509 | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072 | 0.2865    | 0.4332   |](#-pidnet-best-aug-03509--05306----02683--03407--04896--01072--02865-----04332---)
    - [DACS Experiments](#dacs-experiments)
  - [| PIDNet Best Aug| 0.3509 | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072 | 0.2865    | 0.4332   |](#-pidnet-best-aug-03509--05306----02683--03407--04896--01072--02865-----04332----1)
    - [CycleGAN Experiments](#cyclegan-experiments)
    - [PEM Experiments](#pem-experiments)
  - [Main Findings](#main-findings)
  - [References](#references)

---

## Introduction

Semantic segmentation is a critical task in computer vision that involves partitioning an image into semantically meaningful regions for applications such as autonomous driving and remote sensing. However, domain shifts—differences in data distributions between source and target domains—pose significant challenges to model generalization.

This project focuses on real-time domain adaptation for semantic segmentation using the LoveDA dataset, which includes urban and rural environments. The study evaluates various models and techniques to improve segmentation performance across domains while maintaining computational efficiency.

---

## Project Structure

The repository is organized as follows:

- **`/code`**: Implementation of models and techniques.
  - **`/deeplabv2`**: DeepLabV2 implementation.
  - **`/pidnet`**: PIDNet implementation with modifications for domain adaptation.
  - **`/dacs`**: Implementation of Domain Adaptation via Cross-domain Mixed Sampling.
  - **`/cyclegan`**: Code for unpaired image-to-image translation using CycleGAN.
  - **`/pem`**: Prototype-based Efficient MaskFormer implementation.
- **`/experiments`**: Scripts to run experiments with different configurations.
- **`/results`**: Output results, tables, and visualizations.
- **`/data`**: Dataset preprocessing utilities.
- **README.md**: This file.

---

## Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/Martons00/Real-time-Domain-Adaptation-in-Semantic-Segmentation
    cd Real-time-Domain-Adaptation-in-Semantic-Segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

- Python 3.8+
- PyTorch 1.10+ (with CUDA support for GPU training)
- Libraries: NumPy, OpenCV, SciPy, scikit-learn, matplotlib

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### Running Models

**DeepLabV2**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ywc1VuXIAH3tmSfRn8ev3yvSJDGAvSxF?usp=sharing)

Key features:
- Atrous convolutions for dense feature extraction
- ASPP module for multi-scale context

**PIDNet**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/126h9tjDoQ4w1jrmareDz9scs8UbT5VMe?usp=sharing)

Local installation:
```bash
cd PIDNet
pip install -r requirements.txt
python tools/importDataset.py
```
Training commands are available in `run.sh`

Key features:
- Triple-branch architecture (P/I/D)
- Boundary-aware loss function
- Real-time inference capabilities

**CycleGAN**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1coAv3KDPPzsrPN3k-t6qIDQ5AKYw-kEP?usp=sharing)

Dataset: [LoveDa-Urban images](https://zenodo.org/records/14739456)

Key features:
- Unpaired image-to-image translation
- Cycle-consistency loss
- Semantic-guided style transfer

**PEM**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KbzvDoGrSK90cJrAZCP5IxDz6apq-0DR?usp=sharing)

Key features:
- Deformable transformer architecture
- Prototype-based cross-attention
- Panoptic segmentation support

---

## Experiments and Results

### DeepLabV2 Experiments

DeepLabV2 was evaluated on the LoveDA dataset with different configurations of optimizers (Adam, SGD), loss functions , and learning rates.

DeepLab V2 on LoveDA-Urban (Train & Val) , 20 epochs
 
| Number | Optimizer | Loss  | Scheduler | Picture Size | bestmIoU | lr   | Latency (s) |
|--------|-----------|-------|-----------|--------------|---------|------|-------------|
| 1      | Adam      | CE    | True      | 720x720      | 0.2734  | 0.001| 0.005340    |
| 2      | Adam      | DICE  | True      | 720x720      | 0.1274  | 0.001| 0.005236    |
| 3      | Adam      | FOCAL | True      | 720x720      | 0.2559  | 0.001| 0.005734    |
| 4      | Adam      | OCE   | True      | 720x720      | 0.2687  | 0.001| 0.005383    |
| 5      | SGD       | CE    | True      | 720x720      | 0.3364  | 0.01 | 0.005062    |
| 6      | SGD       | DICE  | True      | 720x720      | 0.3112  | 0.01 | 0.005163    |
| 7      | SGD       | FOCAL | True      | 720x720      | 0.2761  | 0.01 | 0.005105    |
| 8      | SGD       | OCE   | True      | 720x720      | 0.3473  | 0.01 | 0.004744    |
| 9      | SGD       | DICE  | True      | 720x720      | 0.3610  | 0.001| 0.004934    |
| 10     | SGD       | CE    | True      | 720x720      | 0.3526  | 0.001| 0.005232    |
| 11     | SGD       | OCE   | True      | 720x720      | 0.3422  | 0.001| 0.005422    |



---

### PIDNet Experiments

PIDNet was tested with various optimizers, loss functions (OHEM, Focal Loss), and resolutions.

Here's the table without the last two columns for PIDNet on LoveDA-Urban (Train & Val), 20 epochs:

| Number | Optimizer | Loss  | Scheduler | Picture Size | mIoU   | Latency (sec) |
|--------|-----------|-------|-----------|--------------|--------|---------------|
| 1      | Adam      | CE    | False     | 720x720      | 0.3617 | 0.029         |
| 2      | Adam      | CE    | False     | 1024x1024    | 0.3906 | 0.027         |
| 3      | Adam      | CE    | True      | 720x720      | 0.3727 | 0.029         |
| 4      | Adam      | CE    | True      | 1024x1024    | 0.3893 | 0.027         |
| 5      | Adam      | OHEM  | False     | 720x720      | 0.3318 | 0.034         |
| 6      | Adam      | OHEM  | True      | 1024x1024    | 0.4275 | 0.033         |
| 7      | Adam      | OHEM  | True      | 720x720      | 0.4368 | 0.030         |
| 8      | Adam      | DICE  | True      | 720x720      | 0.3663 | 0.033         |
| 9      | Adam      | FOCAL | True      | 720x720      | 0.4233 | 0.033         |
| 10     | SDG       | OHEM  | False     | 720x720      | 0.3868 | 0.035         |
| 11     | SDG       | OHEM  | False     | 1024x1024    | 0.3059 | 0.031         |
| 12     | SDG       | CE    | False     | 720x720      | 0.2630 | 0.029         |
| 13     | SDG       | OHEM  | True      | 720x720      | 0.3657 | 0.033         |
| 14     | SDG       | DICE  | False     | 720x720      | 0.3442 | 0.033         |
| 15     | SDG       | FOCAL | False     | 720x720      | 0.2245 | 0.033         |
| 16     | SDG       | CE    | True      | 1024x1024    | 0.3554 | 0.027         |

Adam with OHEM Loss at $$720 \times 720$$ resolution delivered the best performance (mIoU = 0.4368).

---

### Data Augmentation Experiments

Four augmentation techniques were tested to improve cross-domain performance:

| Number | AUG_CHANCE | AUG1  | AUG2  | AUG3  | mIoU | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|--------|------------|-------|-------|-------|---------------|----------|--------|--------|--------|---------|-----------|----------|
| Default| False      | False | False | False | 0.2296        | 0.4158   | 0.2176 | 0.1666 | 0.3349 | 0.0590  | 0.1415    | 0.2716   |
| 1      | TRUE       | False | False | False | 0.2951        | 0.5217   | 0.3381 | 0.3098 | 0.3188 | 0.0673  | 0.0839    | 0.4262   |
| 2      | TRUE       | True  | False | False | 0.3042        | 0.5255   | 0.3789 | 0.3074 | 0.4121 | 0.0377  | 0.0265    | 0.4417   |
| 3      | TRUE       | False | True  | False | 0.3108        | 0.4900   | 0.3403 | 0.3097 | 0.4075 | 0.0582  | 0.1526    | 0.4170   |
| 4      | TRUE       | True  | True  | False | 0.3143        | 0.4766   | 0.3495 | 0.3304 | 0.3810 | 0.0682  | 0.1779    | 0.4165   |
| 5      | TRUE       | False | False | True  | 0.3020        | 0.5257   | 0.3998 | 0.2933 | 0.3413 | 0.0708  | 0.0574    | 0.4257   |
| 6      | TRUE       | True  | False | True  | 0.3008        | 0.5102   | 0.3952 | 0.3130 | 0.3587 | 0.0457  | 0.0505    | 0.4324   |
| 7      | TRUE       | False | True  | True  | 0.3509        | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072  | 0.2865    | 0.4332   |
| 8      | TRUE       | True  | True  | True  | 0.3014        | 0.4877   | 0.3868 | 0.3008 | 0.3700 | 0.0586  | 0.1589    | 0.3472   |

Combining color shifts (AUG2) and geometric transformations (AUG3) yielded the highest mIoU of 0.3509.

---

### Adversarial Learning Experiments

Adversarial learning was applied to PIDNet but resulted in reduced performance compared to baseline methods:

| Model         | mIoU   | Building | Road   | Water  | Barren | Forest | Grassland | Farmland |
|--------------|--------|----------|--------|--------|--------|--------|-----------|----------|
| Baseline     | 0.2296 | 0.4158   | 0.2176 | 0.1666 | 0.3349 | 0.0590 | 0.1415    | 0.2716   |
| PIDNet ADV   | 0.2770 | 0.5145   | 0.2651 | 0.2679 | 0.3808 | 0.1306 | 0.0585    | 0.3217   |
| PIDNet Best Aug| 0.3509 | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072 | 0.2865    | 0.4332   |
---

### DACS Experiments

DACS improved segmentation by leveraging pseudo-labels but underperformed compared to optimal augmentation:

| Model         | mIoU   | Building | Road   | Water  | Barren | Forest | Grassland | Farmland |
|--------------|--------|----------|--------|--------|--------|--------|-----------|----------|
| Baseline     | 0.2296 | 0.4158   | 0.2176 | 0.1666 | 0.3349 | 0.0590 | 0.1415    | 0.2716   |
| PIDNet  DACS          | 0.2918 | 0.5454   | 0.3345 | 0.2913 | 0.4343 | 0.1016  | 0.2310    | 0.3959  |
| PIDNet Best Aug| 0.3509 | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072 | 0.2865    | 0.4332   |
---

### CycleGAN Experiments

CycleGAN-generated images were used to train PIDNet but failed to significantly improve performance:

| Model     | Training Set          | Target Set     | Test Set             | mIoU   | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|--------------|-----------------------|----------------|----------------------|--------|----------|--------|--------|--------|---------|-----------|----------|
| PIDNet       | CycleGAN LoveDa-Urban | LoveDa-Rural   | CycleGAN LoveDa-Urban| 0.4035 | 0.3508   | 0.4845 | 0.5442 | 0.6434 | 0.0919  | 0.3659    | 0.3440   |
| PIDNet        | CycleGAN LoveDa-Urban | LoveDa-Rural   | LoveDa-Rural         | 0.2880 | 0.5127   | 0.1962 | 0.3027 | 0.4716 | 0.0625  | 0.0353    | 0.4349   |

CycleGAN introduced biases that limited its effectiveness for domain adaptation.

---

### PEM Experiments

PEM outperformed other approaches in cross-domain scenarios:

| Codice | Modello               | Training Set          | Test Set          | mIoU    | fwIoU   | mACC    | pACC    |
|--------|-----------------------|-----------------------|-------------------|---------|---------|---------|---------|
| 01     | PEM-URBAN             | LoveDa-Urban          | LoveDa-Urban      | 64.4429 | 60.3522 | 75.1795 | 74.5604 |
| 02     | PEM-RURAL             | LoveDa-Rural          | LoveDa-Rural      | 44.5885 | 56.3823 | 54.6906 | 71.6951 |
| 03     | PEM-CycleGAN-RURAL    | CycleGAN LoveDa-Urban | LoveDa-Rural      | 46.8514 | 54.9652 | 62.2863 | 68.8383 |

PEM demonstrated the potential of transformer-based architectures for domain adaptation.

---

## Main Findings

1. **Best Model Configuration**: For single-domain segmentation, PIDNet with Adam optimizer and OHEM loss achieved the highest mIoU (0.4368).
2. **Domain Adaptation Performance**:
   - Data augmentation (AUG2 + AUG3) provided the best cross-domain results (mIoU = 0.3509).
   - PEM outperformed other approaches in domain adaptation tasks (mIoU = 0.4685).
3. **Limitations of CycleGAN and Adversarial Learning**:
   - CycleGAN-generated images introduced biases that reduced segmentation accuracy.
   - Adversarial learning did not yield significant improvements on the LoveDA dataset.

---

## References

1. Ianniello et al., *Real-time Domain Adaptation in Semantic Segmentation*, January 2025.
2. Chen et al., *DeepLab: Semantic Image Segmentation*, IEEE Transactions on PAMI, 2017.
3. Feng et al., *PIDNet: A Real-Time Semantic Segmentation Network*, arXiv preprint, March 2021.
4. Zhu et al., *Unpaired Image-to-Image Translation Using CycleGAN*, arXiv preprint, March 2020.

For more details, refer to the paper or contact the authors via GitHub Issues or Pull Requests!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/45403030/39671f9a-bd2e-4bd4-a3ae-fef967684b9a/s327313_s324807_s326811_project4.pdf