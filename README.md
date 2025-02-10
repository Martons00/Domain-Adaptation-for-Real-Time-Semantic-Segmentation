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
    - [DACS Experiments](#dacs-experiments)
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
   git clone https://github.com/LucaIanniello/AML2024.git
   cd AML2024
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure paths and hyperparameters in `config.yaml`.

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

- **DeepLabV2 Training**:
  ```bash
  python code/deeplabv2/train.py --config config/deeplabv2.yaml
  ```
- **PIDNet Training**:
  ```bash
  python code/pidnet/train.py --config config/pidnet.yaml
  ```
- **Data Augmentation Experiments**:
  ```bash
  python experiments/augmentation/run_augmentation.py --config config/augmentation.yaml
  ```
- **Adversarial Learning**:
  ```bash
  python experiments/adversarial/run_adversarial.py --config config/adversarial.yaml
  ```
- **CycleGAN Training**:
  ```bash
  python code/cyclegan/train.py --config config/cyclegan.yaml
  ```

---

## Experiments and Results

### DeepLabV2 Experiments

DeepLabV2 was evaluated on the LoveDA dataset with different configurations of optimizers (Adam, SGD), loss functions (Cross Entropy, Dice Loss), and learning rates.

| Optimizer | Loss Function | mIoU | Learning Rate |
|-----------|---------------|------|---------------|
| Adam      | CE            | 0.2734 | 0.001       |
| SGD       | Dice          | 0.3610 | 0.001       |

SGD with Dice Loss achieved the best performance (mIoU = 0.3610).

---

### PIDNet Experiments

PIDNet was tested with various optimizers, loss functions (OHEM, Focal Loss), and resolutions.

| Optimizer | Loss Function | Resolution | mIoU |
|-----------|---------------|------------|------|
| Adam      | OHEM          | $$720 \times 720$$ | 0.4368 |
| SGD       | Focal         | $$720 \times 720$$ | 0.2245 |

Adam with OHEM Loss at $$720 \times 720$$ resolution delivered the best performance (mIoU = 0.4368).

---

### Data Augmentation Experiments

Four augmentation techniques were tested to improve cross-domain performance:

| Augmentation Technique    | mIoU |
|---------------------------|------|
| Baseline                 | 0.2296 |
| AUG_CHANCE               | 0.2951 |
| AUG2 + AUG3              | **0.3509** |

Combining color shifts (AUG2) and geometric transformations (AUG3) yielded the highest mIoU of 0.3509.

---

### Adversarial Learning Experiments

Adversarial learning was applied to PIDNet but resulted in reduced performance compared to baseline methods:

| Model         | mIoU |
|---------------|------|
| PIDNet        | **0.3509** |
| PIDNet ADV    | 0.2770 |

---

### DACS Experiments

DACS improved segmentation by leveraging pseudo-labels but underperformed compared to optimal augmentation:

| Model         | mIoU |
|---------------|------|
| PIDNet        | **0.3509** |
| PIDNet DACS   | 0.2918 |

---

### CycleGAN Experiments

CycleGAN-generated images were used to train PIDNet but failed to significantly improve performance:

| Model         | Training Set         | Test Set         | mIoU |
|---------------|----------------------|------------------|------|
| PIDNet        | CycleGAN Urban       | Rural            | 0.2880 |

CycleGAN introduced biases that limited its effectiveness for domain adaptation.

---

### PEM Experiments

PEM outperformed other approaches in cross-domain scenarios:

| Model         | Training Set         | Test Set         | mIoU |
|---------------|----------------------|------------------|------|
| PEM-CycleGAN  | CycleGAN Urban       | Rural            | **0.4685** |

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