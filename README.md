# Tri-KA: Tri-level Knowledge Anchoring Test-Time Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the **Tri-KA** framework, as detailed in our MICCAI 2026 paper. Tri-KA is a privacy-preserving Test-Time Adaptation (TTA) method designed to tackle severe domain shifts in cross-site 3D medical image segmentation (e.g., highly anisotropic sagittal rectal cancer MRI).

## 💡 Overview
Generic TTA methods often cause semantic drift and structural distortion when applied to 3D medical networks like nnU-Net. To address this under strict patient privacy constraints, Tri-KA safely distills and transmits highly compressed, non-identifiable source priors across three dimensions (Input-Level Style, Feature-Level Semantics, Output-Level Distribution). During deployment, it leverages a novel **Deep-to-Shallow** guidance mechanism to adapt to target domain textures while securely anchoring invariant anatomical topologies.

## ⚙️ Prerequisites
Ensure your environment satisfies the following dependencies:
* Python 3.8+
* PyTorch 2.0+
* [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
* SimpleITK, medpy, scikit-learn, pandas

Install the core dependencies via:
```bash
pip install -r requirements.txt
