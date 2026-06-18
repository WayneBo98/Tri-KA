Tri-KA: Tri-level Knowledge Anchoring Test-Time Adaptation
This repository contains the official PyTorch implementation of the Tri-KA framework, as detailed in our MICCAI 2026 paper. Tri-KA is a privacy-preserving Test-Time Adaptation (TTA) method designed to tackle severe domain shifts in cross-site 3D medical image segmentation (e.g., highly anisotropic sagittal rectal cancer MRI).

💡 Overview
Generic TTA methods often cause semantic drift and structural distortion when applied to 3D medical networks like nnU-Net. To address this under strict patient privacy constraints, Tri-KA safely distills and transmits highly compressed, non-identifiable source priors across three dimensions (Input-Level Style, Feature-Level Semantics, Output-Level Distribution). During deployment, it leverages a novel Deep-to-Shallow guidance mechanism to adapt to target domain textures while securely anchoring invariant anatomical topologies.

⚙️ Prerequisites
Ensure your environment satisfies the following dependencies:

Python 3.8+

PyTorch 2.0+

nnU-Net v2

SimpleITK, medpy, scikit-learn, pandas

🚀 Quick Start Pipeline
Our pipeline is modularized into four sequential steps. Please ensure your nnUNet_raw, nnUNet_preprocessed, and nnUNet_results environment variables are properly configured before proceeding.

Step 1: Extract Source Priors (Offline)
Extract the required privacy-preserving statistical priors (Class Distribution, FFT Style, and Deep Semantic Prototypes) from your source domain training data. This step only needs to be run once.

Bash
python extract_source_info.py \
    --base_nnunet_dir /path/to/your/nnUNet/DATASET \
    --model_folder DatasetXXX_YourTask/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --train_img_dir /path/to/source/imagesTr \
    --train_label_dir /path/to/source/labelsTr \
    --num_classes 3 \
    --save_dir ./source_priors
Step 2: Run Baseline Inference (Optional)
Evaluate the unadapted model directly on the unseen target domain to establish your pre-TTA baseline performance.

Bash
python run_baseline.py \
    --fold 0 \
    --gpu 0 \
    --base_nnunet_dir /path/to/your/nnUNet/DATASET \
    --model_folder DatasetXXX_YourTask/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --target_images_dir /path/to/target/imagesTs \
    --output_dir ./results/baseline_inference
Step 3: Run Tri-KA Test-Time Adaptation (Online)
Execute the core Tri-KA adaptation process. This script dynamically loads the extracted priors and performs robust test-time optimization on the target cases.

Bash
python run_trika.py \
    --fold 0 \
    --gpu 0 \
    --base_nnunet_dir /path/to/your/nnUNet/DATASET \
    --model_folder DatasetXXX_YourTask/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --target_images_dir /path/to/target/imagesTs \
    --source_priors_dir ./source_priors \
    --output_dir ./results/tri_ka_adapted
Step 4: Evaluate Metrics
Calculate standard clinical metrics (Dice, ASD) with built-in Largest Connected Component (LCC) post-processing to filter macroscopic hallucinations.

Bash
python evaluation.py \
    --tta_root ./results/tri_ka_adapted \
    --gt_dir /path/to/target/labelsTs \
    --output_dir ./evaluation_results
📝 Citation
If you find this code or our conceptual framework useful in your research, please consider citing our paper:

代码段
@inproceedings{wang2026trika,
  title={Tri-KA: Tri-level Knowledge Anchoring Test-Time Adaptation for Source-Free Cross-Site MRI Rectal Cancer Segmentation},
  author={Wang, Bo and others},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year={2026},
  publisher={Springer Nature Switzerland}
}
