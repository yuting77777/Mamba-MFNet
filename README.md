# Mamba‑MFNet
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Mamba‑MFNet** is a novel multi‑modal medical image fusion network based on state space models (Mamba).  
> It effectively preserves complementary information while eliminating redundancy, achieving state‑of‑the‑art performance on CT‑MRI, PET‑MRI, and SPECT‑MRI fusion tasks.

---

## 📌 Overview

Mamba‑MFNet consists of four main stages:

1. **Data Preprocessing** – RGB‑to‑YUV conversion to separate luminance and chrominance.
2. **Feature Extraction** – Auxiliary Augmented Feature Extraction (AAFE) + Dual‑Branch Mamba Blocks (DBMB) for local details and long‑range dependencies.
3. **Feature Fusion** – Cross‑modal Multi‑level Mamba Fusion (CM3F) module for deep cross‑modal interaction.
4. **Image Reconstruction** – Progressive decoding and inverse YUV transform to obtain the final RGB fused image.

The network is trained with a composite loss (intensity + gradient + SSIM) to balance structure, texture, and visual fidelity.

> 📄 Full details will be available in our upcoming paper. This repository provides the code, training/testing instructions, and pretrained models.

---

## 🗂️ Dataset

All data are taken from the **AANLIB database** of Harvard Medical School:  
[https://www.med.harvard.edu/AANLIB/home.html](https://www.med.harvard.edu/AANLIB/home.html)

We provide a ready‑to‑use version of the dataset (split into training/test sets) at:  
[https://github.com/yuting77777/AANLIB-database-of-Harvard-Medical-School.git](https://github.com/yuting77777/AANLIB-database-of-Harvard-Medical-School.git)

The dataset contains:
- **184 CT‑MRI pairs**
- **269 PET‑MRI pairs**
- **357 SPECT‑MRI pairs**

Images are randomly split 8:2 into training / test. During training, overlapping cropping to `128×128` patches is applied for data augmentation.

---

## ⚙️ Installation

### 1. Create a conda environment

```bash
conda create -n mamba_mfnet python=3.8
conda activate mamba_mfnet
