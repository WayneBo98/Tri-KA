import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Run Baseline Direct Inference (Without TTA)")
    
    # --- System & Environment ---
    parser.add_argument('-f', '--fold', type=int, required=True, help='Fold number (0-4)')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU device ID')
    
    # --- Paths (Replace defaults with your own generic paths) ---
    parser.add_argument('--base_nnunet_dir', type=str, default="/path/to/your/nnUNet/DATASET",
                        help="Base directory for nnUNet.")
    parser.add_argument('--model_folder', type=str, default="DatasetXXX_YourTask/nnUNetTrainer__nnUNetPlans__3d_fullres",
                        help="Relative path to the trained nnUNet model.")
    parser.add_argument('--target_images_dir', type=str, default="/path/to/target_domain/imagesTs",
                        help="Directory containing target domain images to predict.")
    parser.add_argument('--output_dir', type=str, default="./results/baseline_inference",
                        help="Directory to save baseline predictions.")
    
    return parser.parse_args()

args = parse_args()

# ================= 1. Environment & Setup =================
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['nnUNet_raw'] = os.path.join(args.base_nnunet_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(args.base_nnunet_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(args.base_nnunet_dir, 'nnUNet_trained_models')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

CURRENT_FOLD = args.fold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NNUNET_MODEL_FOLDER = os.path.join(os.environ['nnUNet_results'], args.model_folder)

OUTPUT_DIR = os.path.join(args.output_dir, f"fold_{CURRENT_FOLD}", "predict") 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

print(f"🚀 Running Baseline Direct Inference on FOLD {CURRENT_FOLD}")
print(f"📂 Output Dir: {OUTPUT_DIR}")

# ================= 2. Utility Functions =================
def get_divisors_from_plans(plans_manager):
    """
    Extract downsampling factors accurately from plans.json.
    """
    configuration = plans_manager.get_configuration("3d_fullres")
    pool_op_kernel_sizes = configuration.pool_op_kernel_sizes
    
    div_d, div_h, div_w = 1, 1, 1
    for kernel in pool_op_kernel_sizes:
        div_d *= kernel[0]
        div_h *= kernel[1]
        div_w *= kernel[2]
        
    return (div_d, div_h, div_w)

def pad_to_compatible_size(input_tensor, plans_manager):
    div_d, div_h, div_w = get_divisors_from_plans(plans_manager)
    
    d, h, w = input_tensor.shape[2:]
    target_d = (d + div_d - 1) // div_d * div_d
    target_h = (h + div_h - 1) // div_h * div_h
    target_w = (w + div_w - 1) // div_w * div_w
    pd, ph, pw = target_d - d, target_h - h, target_w - w
    
    if pd == 0 and ph == 0 and pw == 0:
        return input_tensor, (d, h, w)
        
    min_val = input_tensor.min()
    padded_tensor = F.pad(input_tensor, (0, pw, 0, ph, 0, pd), mode='constant', value=min_val)
    return padded_tensor, (d, h, w)

# ================= 3. Initialize nnU-Net =================
print("Initializing nnU-Net Predictor...")
predictor = nnUNetPredictor(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=DEVICE, verbose=False, allow_tqdm=True
)

try:
    predictor.initialize_from_trained_model_folder(
        NNUNET_MODEL_FOLDER, 
        use_folds=(CURRENT_FOLD,), 
        checkpoint_name='checkpoint_best.pth'
    )
except Exception as e:
    print(f"❌ Error loading Fold {CURRENT_FOLD}: {e}")
    exit(1)

preprocessor_class = predictor.configuration_manager.preprocessor_class
predictor.preprocessor = preprocessor_class(verbose=predictor.verbose_preprocessing)
net = predictor.network

# Set network to evaluation mode
net.eval()

# ================= 4. Inference Loop =================
test_files = [f for f in os.listdir(args.target_images_dir) if f.endswith(".nii.gz")]
print(f"\nFound {len(test_files)} test cases. Starting Direct Inference...")

for case_file in test_files:
    save_path = os.path.join(OUTPUT_DIR, case_file)
    if os.path.exists(save_path):
        print(f"Skipping {case_file} (Already predicted)")
        continue

    print(f"Processing {case_file} ...")
    
    img_path = os.path.join(args.target_images_dir, case_file)
    img_sitk = sitk.ReadImage(img_path) 
    
    # --- A. Preprocessing ---
    data, seg, data_properties = predictor.preprocessor.run_case(
        [img_path], None, predictor.plans_manager, 
        predictor.configuration_manager, predictor.dataset_json
    )
    
    input_tensor_raw = torch.from_numpy(data).to(DEVICE).unsqueeze(0)
    input_tensor, original_shape = pad_to_compatible_size(input_tensor_raw, predictor.plans_manager)
    
    # --- B. Direct Inference (Single Forward Pass) ---
    with torch.no_grad():
        final_output = net(input_tensor)
        final_pred_padded = torch.argmax(final_output, dim=1)
        
        # --- C. Post-processing ---
        od, oh, ow = original_shape
        final_pred = final_pred_padded[0, :od, :oh, :ow]

        # 1. Transpose back to original axes
        if 'transpose_forward' in data_properties:
            forward_order = data_properties['transpose_forward']
            inverse_order = np.argsort(forward_order).tolist()
            final_pred = final_pred.permute(*inverse_order)
        
        final_arr = final_pred.cpu().numpy().astype(np.uint8)

        # 2. Undo Resampling (Nearest Neighbor for discrete labels)
        if 'shape_after_cropping_and_before_resampling' in data_properties:
            target_shape_after_crop = tuple(data_properties['shape_after_cropping_and_before_resampling'])
            if tuple(final_arr.shape) != target_shape_after_crop:
                final_tensor = torch.from_numpy(final_arr).unsqueeze(0).unsqueeze(0).float()
                final_restored = F.interpolate(final_tensor, size=target_shape_after_crop, mode='nearest')
                final_arr = final_restored.squeeze().cpu().numpy().astype(np.uint8)

        # 3. Undo Cropping (Paste back to original canvas)
        if 'bbox_used_for_cropping' in data_properties:
            bbox = data_properties['bbox_used_for_cropping']
            shape_original = tuple(data_properties['shape_before_cropping'])
            
            full_arr = np.zeros(shape_original, dtype=final_arr.dtype)
            slicer = (
                slice(bbox[0][0], bbox[0][1]),
                slice(bbox[1][0], bbox[1][1]),
                slice(bbox[2][0], bbox[2][1])
            )
            try:
                full_arr[slicer] = final_arr
                final_arr = full_arr
            except Exception as e:
                print(f"⚠️ Warning: Uncrop failed ({e}). Fallback to resize.")
                target_size_zyx = img_sitk.GetSize()[::-1]
                final_tensor = torch.from_numpy(final_arr).unsqueeze(0).unsqueeze(0).float()
                final_restored = F.interpolate(final_tensor, size=target_size_zyx, mode='nearest')
                final_arr = final_restored.squeeze().cpu().numpy().astype(np.uint8)

        # --- D. Save Prediction ---
        final_img = sitk.GetImageFromArray(final_arr)
        final_img.CopyInformation(img_sitk)
        sitk.WriteImage(final_img, save_path)
        print(f"✅ Saved Baseline Prediction: {save_path}")

print(f"\n🎉 Baseline Inference for Fold {CURRENT_FOLD} completed!")