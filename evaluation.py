import os
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import dc, asd
from scipy.ndimage import label 
import warnings

# Suppress annoying warnings from medpy if any
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dice and ASD metrics for TTA results with LCC post-processing.")
    
    # --- Paths ---
    parser.add_argument('--tta_root', type=str, default="./results/tri_ka_adapted",
                        help="Root directory containing the TTA fold results (e.g., fold_0, fold_1).")
    parser.add_argument('--gt_dir', type=str, default="/path/to/your/labelsTs",
                        help="Directory containing ground truth labels.")
    parser.add_argument('--output_dir', type=str, default="./evaluation_results",
                        help="Directory to save the CSV and summary TXT reports.")
    
    # --- Parameters ---
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds to evaluate.")
    parser.add_argument('--penalty_asd', type=float, default=100.0, 
                        help="Penalty value for ASD when prediction or GT is entirely empty.")
    
    return parser.parse_args()

args = parse_args()

# ================= 1. Utility Functions =================

def get_image_and_spacing(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing() 
    return arr, spacing[::-1] # Convert from SimpleITK (x,y,z) to numpy (z,y,x)

def keep_largest_connected_component(mask):
    """
    Extracts the Largest Connected Component (LCC) from a binary mask.
    Effectively filters out isolated false positive noise.
    """
    labels, num_features = label(mask)
    if num_features == 0:
        return mask # All background, return directly
        
    counts = np.bincount(labels.flat)
    if len(counts) <= 1:
        return mask # Only background
        
    # Find the largest foreground component (skip index 0 which is background)
    largest_cc_idx = np.argmax(counts[1:]) + 1
    return (labels == largest_cc_idx).astype(np.uint8)

def calculate_metrics(pred_arr, gt_arr, spacing, class_id, penalty_asd=100.0, use_lcc=True):
    """
    Calculates Dice and Average Surface Distance (ASD) for a specific class.
    """
    p = (pred_arr == class_id).astype(np.uint8)
    g = (gt_arr == class_id).astype(np.uint8)

    # 🚀 Core operation: Apply LCC post-processing to prediction
    if use_lcc:
        p = keep_largest_connected_component(p)

    intersection = (p * g).sum()
    sum_p = p.sum()
    sum_g = g.sum()
    
    # --- Dice Calculation ---
    if sum_p == 0 and sum_g == 0: 
        dice = 1.0
    elif sum_p == 0 or sum_g == 0: 
        dice = 0.0
    else: 
        dice = 2.0 * intersection / (sum_p + sum_g)

    # --- ASD Calculation ---
    if sum_p == 0 and sum_g == 0:
        asd_val = 0.0
    elif sum_p == 0 or sum_g == 0:
        asd_val = penalty_asd
    else:
        try:
            asd_val = asd(p, g, voxelspacing=spacing)
        except:
            asd_val = penalty_asd

    return dice, asd_val

def find_gt_path(case_name, gt_folder):
    """
    Robustly matches prediction filenames with Ground Truth filenames.
    Handles nnU-Net's '_0000' suffix conventions if present.
    """
    path = os.path.join(gt_folder, case_name)
    if os.path.exists(path): return path
    
    path = os.path.join(gt_folder, case_name.replace(".nii.gz", "_0000.nii.gz"))
    if os.path.exists(path): return path
    
    if "_0000.nii.gz" in case_name:
        clean_name = case_name.replace("_0000.nii.gz", ".nii.gz")
        path = os.path.join(gt_folder, clean_name)
        if os.path.exists(path): return path
        
    return None

# ================= 2. Main Evaluation Pipeline =================

def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_csv = os.path.join(args.output_dir, "final_cv_metrics_case_level.csv")
    output_txt = os.path.join(args.output_dir, "final_cv_summary_report.txt")

    all_results = []
    print(f"🚀 Starting {args.num_folds}-Fold Evaluation (with LCC Post-processing)...")
    
    for fold in range(args.num_folds):
        fold_dir = os.path.join(args.tta_root, f"fold_{fold}", "predict")
        if not os.path.exists(fold_dir):
            print(f"⚠️ Warning: Directory not found: {fold_dir}")
            continue
            
        pred_files = sorted([f for f in os.listdir(fold_dir) if f.endswith(".nii.gz")])
        print(f"   Processing Fold {fold} ({len(pred_files)} cases)...")
        
        for pred_file in pred_files:
            pred_path = os.path.join(fold_dir, pred_file)
            gt_path = find_gt_path(pred_file, args.gt_dir)
            if gt_path is None: 
                print(f"     [Skip] GT not found for {pred_file}")
                continue
                
            try:
                pred_arr, spacing = get_image_and_spacing(pred_path)
                gt_arr, gt_spacing = get_image_and_spacing(gt_path)
            except Exception as e: 
                print(f"     [Error] Reading {pred_file}: {e}")
                continue
                
            if pred_arr.shape != gt_arr.shape: 
                print(f"     [Mismatch] Shape mismatch for {pred_file}")
                continue
                
            # Evaluate Class 1 (Rectum Wall) and Class 2 (Tumor)
            d_wall, a_wall = calculate_metrics(pred_arr, gt_arr, spacing, class_id=1, penalty_asd=args.penalty_asd)
            d_tumor, a_tumor = calculate_metrics(pred_arr, gt_arr, spacing, class_id=2, penalty_asd=args.penalty_asd)
            
            all_results.append({
                "Fold": fold,
                "Case": pred_file,
                "Wall_Dice": d_wall, "Wall_ASD": a_wall,
                "Tumor_Dice": d_tumor, "Tumor_ASD": a_tumor
            })

    if len(all_results) == 0: 
        print("\n❌ No valid results to summarize. Check your paths.")
        return

    # --- Save Case-Level CSV ---
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Case-level data saved to: {output_csv}")

    # --- Generate Summary Report ---
    metrics_cols = ["Wall_Dice", "Wall_ASD", "Tumor_Dice", "Tumor_ASD"]
    
    # Calculate means and standard deviations across folds
    fold_means = df.groupby("Fold")[metrics_cols].mean()
    cv_mean = fold_means.mean()
    cv_std = fold_means.std()

    lines = []
    lines.append("="*60)
    lines.append(f" {args.num_folds}-FOLD CV REPORT (With Largest Connected Component)")
    lines.append("="*60)
    lines.append(f"Method: Average of {len(fold_means)} folds")
    lines.append("-" * 60)
    
    def fmt_percent(col):
        # Format Dice as percentage (e.g., 61.53 ± 1.92) to match standard paper formats
        return f"{cv_mean[col]*100:.2f}% ± {cv_std[col]*100:.2f}%"

    def fmt_dist(col):
        # Format ASD as float
        return f"{cv_mean[col]:.2f} ± {cv_std[col]:.2f}"

    lines.append(f"[Rectum Wall (Anatomical Anchor - Class 1)]")
    lines.append(f"  Dice: {fmt_percent('Wall_Dice')}")
    lines.append(f"  ASD : {fmt_dist('Wall_ASD')} mm")
    lines.append("-" * 30)

    lines.append(f"[Tumor (Clinical Target - Class 2)]")
    lines.append(f"  Dice: {fmt_percent('Tumor_Dice')}")
    lines.append(f"  ASD : {fmt_dist('Tumor_ASD')} mm")
    
    lines.append("\n" + "="*60)
    lines.append(" Breakdown by Fold")
    lines.append("="*60)
    lines.append(fold_means.to_string())

    summary_txt = "\n".join(lines)
    print(summary_txt)
    
    with open(output_txt, "w") as f:
        f.write(summary_txt)
    print(f"\n✅ Final Report saved to: {output_txt}")

if __name__ == "__main__":
    main()