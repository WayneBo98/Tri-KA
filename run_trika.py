import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Run Tri-KA Test-Time Adaptation")
    
    # --- System & Environment ---
    parser.add_argument('--fold', type=int, required=True, help='Fold number (0-4)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    # --- Paths (Replace defaults with your own generic paths) ---
    parser.add_argument('--base_nnunet_dir', type=str, default="/path/to/your/nnUNet/DATASET",
                        help="Base directory for nnUNet.")
    parser.add_argument('--model_folder', type=str, default="DatasetXXX_YourTask/nnUNetTrainer__nnUNetPlans__3d_fullres",
                        help="Relative path to the trained nnUNet model.")
    parser.add_argument('--target_images_dir', type=str, default="/path/to/target_domain/imagesTs",
                        help="Directory containing target domain images to adapt.")
    parser.add_argument('--source_priors_dir', type=str, default="./source_priors",
                        help="Directory containing the extracted source priors (.npz, .npy).")
    parser.add_argument('--output_dir', type=str, default="./results/tri_ka_adapted",
                        help="Directory to save adapted predictions.")
    
    # --- TTA Hyperparameters ---
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate for shallow encoder fine-tuning.")
    parser.add_argument('--steps', type=int, default=20, help="Number of adaptation steps per subject.")
    parser.add_argument('--lambda_prior', type=float, default=2.0, help="Weight for the KL distribution prior.")
    parser.add_argument('--fft_beta', type=float, default=0.5, help="Stylization intensity for FFT adaptation.")
    parser.add_argument('--conf_thresh', type=float, default=0.95, help="Confidence threshold for pseudo-labels.")
    
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

OUTPUT_DIR = os.path.join(args.output_dir, f"fold_{CURRENT_FOLD}") 
os.makedirs(os.path.join(OUTPUT_DIR, "predict"), exist_ok=True)

# Dynamically link the priors extracted from the previous script
PRIOR_DIST_PATH = os.path.join(args.source_priors_dir, "source_class_prior.npy")
PROTOTYPES_PATH = os.path.join(args.source_priors_dir, f"source_multi_level_prototypes_fold{CURRENT_FOLD}.npz")
SOURCE_STYLE_PATH = os.path.join(args.source_priors_dir, "source_style_2d.npy") 

# ==============================================================================
# ⚠️ IMPORTANT: nnU-Net Topology Configuration (Deep vs. Shallow) ⚠️
# Depending on the target spacing and patch size, nnU-Net automatically configures 
# a different number of encoder stages (typically 5 to 7). 
# We use negative indices to robustly reference layers from deep to shallow.
#
# In Tri-KA, the division between "Deep" and "Shallow" is strictly based on the 
# physical properties of the network's operations (Anisotropic vs. Isotropic):
#
#   - SHALLOW Layers (Weight = 0.0, e.g., stages -5 to -7): 
#     These stages perform ANISOTROPIC convolutions and pooling to align the highly 
#     anisotropic voxel spacing of the input MRI. They are actively updated during 
#     TTA to adapt to target domain textures.
#
#   - DEEP Layers (Weight = 1.0, e.g., stages -1 to -4): 
#     Once spatial alignment is roughly achieved, these stages begin ISOTROPIC 
#     operations. They are kept frozen to act as domain-invariant semantic anchors 
#     for computing the Prototype Loss.
#
# 👉 USER ACTION: Check your nnU-Net plans. If your architecture has a different 
# transition point between anisotropic and isotropic stages, adjust the LAYER_WEIGHTS 
# below to ensure only isotropic layers receive a weight of 1.0.
# ==============================================================================
TARGET_LAYER_IDX_LIST = [-7, -6, -5, -4, -3, -2, -1]
LAYER_WEIGHTS = {-7: 0.0, -6: 0.0, -5: 0.0, -4: 1.0, -3: 1.0, -2: 1.0, -1: 1.0}
MIN_PIXELS_PER_CLASS = 10 
ANATOMY_KERNEL_SIZE = 5 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

print(f"🚀 Running Tri-KA TTA on FOLD {CURRENT_FOLD}")
print(f"📂 Output Dir: {OUTPUT_DIR}")

# ================= 2. Utility Functions =================
try:
    from skimage.filters import threshold_otsu
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def get_robust_mask(vol):
    mid_slice = vol[vol.shape[0]//2]
    if HAS_SKIMAGE:
        thresh = threshold_otsu(mid_slice)
    else:
        h, w = mid_slice.shape
        corners = np.concatenate([
            mid_slice[0:10, 0:10].flatten(), mid_slice[0:10, w-10:w].flatten(),
            mid_slice[h-10:h, 0:10].flatten(), mid_slice[h-10:h, w-10:w].flatten()
        ])
        thresh = np.mean(corners) + 3 * np.std(corners)
    return vol > thresh

def apply_fourier_adaptation_clean(target_vol_np, source_style_path, beta=0.1, L_ratio=0.01):
    if not os.path.exists(source_style_path): return target_vol_np
    
    D, H, W = target_vol_np.shape
    adapted_vol = np.zeros_like(target_vol_np)
    
    source_data = np.load(source_style_path)
    source_amp_raw = source_data['avg_amp'] if 'avg_amp' in source_data else source_data # Fallback if dict
    src_t = torch.from_numpy(source_amp_raw).unsqueeze(0).unsqueeze(0)
    src_resized = F.interpolate(src_t, size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist_sq = (y - cy)**2 + (x - cx)**2
    sigma = max(H, W) * L_ratio * 2.0 
    gaussian_mask = np.exp(-dist_sq / (2 * sigma**2))
    foreground_mask = get_robust_mask(target_vol_np)

    for d in range(D):
        slice_img = target_vol_np[d]
        fft2 = np.fft.fft2(slice_img)
        amp = np.abs(fft2)
        phase = np.angle(fft2)
        amp_shift = np.fft.fftshift(amp)
        
        mask_energy = (gaussian_mask > 0.01)
        t_energy = np.mean(amp_shift[mask_energy]) + 1e-8
        s_energy = np.mean(src_resized[mask_energy]) + 1e-8
        
        src_aligned = src_resized * (t_energy / s_energy)
        mixing_weight = gaussian_mask * beta
        mixed_amp = amp_shift + mixing_weight * (src_aligned - amp_shift)
        mixed_amp[cy, cx] = amp_shift[cy, cx]  # Preserve DC component
        
        amp_new = np.fft.ifftshift(mixed_amp)
        fft_new = amp_new * np.exp(1j * phase)
        adapted_vol[d] = np.abs(np.fft.ifft2(fft_new))

    final_vol = np.where(foreground_mask, adapted_vol, target_vol_np)
    return final_vol

def dilate_mask(mask, kernel_size=5):
    if mask.dim() == 4: mask = mask.unsqueeze(1)
    padding = kernel_size // 2
    dilated = F.max_pool3d(mask.float(), kernel_size, stride=1, padding=padding)
    return dilated > 0.5

def get_divisors_from_plans(plans_manager):
    configuration = plans_manager.get_configuration("3d_fullres")
    pool_op_kernel_sizes = configuration.pool_op_kernel_sizes
    div_d, div_h, div_w = 1, 1, 1
    for kernel in pool_op_kernel_sizes:
        div_d *= kernel[0]; div_h *= kernel[1]; div_w *= kernel[2]
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
    return F.pad(input_tensor, (0, pw, 0, ph, 0, pd), mode='constant', value=min_val), (d, h, w)

def get_point_features(feat, indices_zyx, target_shape):
    _, C, D, H, W = feat.shape
    pts = indices_zyx[:, [2, 1, 0]].float()
    pts = (pts / (target_shape[[2, 1, 0]] - 1)) * 2 - 1
    pts = pts.view(1, -1, 1, 1, 3)
    sampled_feat = F.grid_sample(feat, pts, mode='bilinear', align_corners=False)
    return sampled_feat.view(C, -1).t()

# ================= 3. Initialize Model & Load Priors =================
if __name__ == "__main__":
    
    # --- Load Output-Level Prior (Class Distribution) ---
    if not os.path.exists(PRIOR_DIST_PATH):
        raise FileNotFoundError(f"Class Prior missing: {PRIOR_DIST_PATH}. Run extract_source_priors.py first.")
    class_prior_np = np.load(PRIOR_DIST_PATH)
    target_prior_dist = torch.tensor(class_prior_np, device=DEVICE).float()
    print(f"✅ Loaded Class Prior Distribution: {class_prior_np}")

    print("\nInitializing nnU-Net Predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=DEVICE, verbose=False, allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        NNUNET_MODEL_FOLDER, use_folds=(CURRENT_FOLD,), checkpoint_name='checkpoint_best.pth'
    )

    preprocessor_class = predictor.configuration_manager.preprocessor_class
    predictor.preprocessor = preprocessor_class(verbose=predictor.verbose_preprocessing)
    net = predictor.network
    net.to(DEVICE)

    # --- Register Hooks for Feature-Level Guidance ---
    captured_features_dict = {}
    def make_hook(layer_name):
        def hook(module, input, output):
            captured_features_dict[layer_name] = output
        return hook
    
    hook_handles = []
    for idx in TARGET_LAYER_IDX_LIST:
        layer_name = f"layer_{idx}"
        hook_handles.append(net.encoder.stages[idx].register_forward_hook(make_hook(layer_name)))

    # --- Load Feature-Level Prototypes ---
    if not os.path.exists(PROTOTYPES_PATH):
        raise FileNotFoundError(f"Prototypes missing: {PROTOTYPES_PATH}")
    npz_file = np.load(PROTOTYPES_PATH)
    multi_level_prototypes = {}
    num_classes = len(class_prior_np)
    for layer_idx in TARGET_LAYER_IDX_LIST:
        multi_level_prototypes[layer_idx] = {}
        for cls_id in range(num_classes):
            key = f"layer_{layer_idx}_class_{cls_id}"
            multi_level_prototypes[layer_idx][cls_id] = torch.from_numpy(npz_file[key]).to(DEVICE)
    print("✅ Prototypes loaded.")

    # Cache original model weights
    initial_net_dict = {k: v.clone() for k, v in net.state_dict().items()}

    # ================= 4. TTA Loop =================
    test_files = [f for f in os.listdir(args.target_images_dir) if f.endswith(".nii.gz")]
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n🚀 Starting Tri-KA TTA on {len(test_files)} target cases...")

    for case_file in test_files:
        save_path = os.path.join(OUTPUT_DIR, "predict", case_file)
        if os.path.exists(save_path):
            print(f"Skipping {case_file} (Already processed)")
            continue

        print(f"\nProcessing {case_file} ...")
        net.load_state_dict(initial_net_dict) # Reset for each subject
        net.train()
        
        # --- Deep-to-Shallow Optimization Mechanism ---
        # We explicitly fine-tune ONLY the shallow layers to safely adapt target textures,
        # while keeping deep layers and the decoder strictly frozen to anchor invariant topologies.
        #
        # 👉 USER ACTION: If your nnU-Net has a different number of stages, adjust the indices 
        # below to match your shallowest stages (aligning with the 0.0 weights in LAYER_WEIGHTS).
        params_to_update = []
        params_to_update.extend(list(net.encoder.stages[-5].parameters()))
        params_to_update.extend(list(net.encoder.stages[-6].parameters()))
        params_to_update.extend(list(net.encoder.stages[-7].parameters()))
        
        optimizer = optim.Adam(params_to_update, lr=args.lr)
        
        img_path = os.path.join(args.target_images_dir, case_file)
        img_sitk = sitk.ReadImage(img_path)
        
        data, _, data_properties = predictor.preprocessor.run_case(
            [img_path], None, predictor.plans_manager, 
            predictor.configuration_manager, predictor.dataset_json
        )
        
        # --- Apply Input-Level Fourier Prior ---
        data[0] = apply_fourier_adaptation_clean(data[0], SOURCE_STYLE_PATH, beta=args.fft_beta, L_ratio=0.02)
        
        input_tensor_raw = torch.from_numpy(data).to(DEVICE).unsqueeze(0)
        input_tensor, original_shape = pad_to_compatible_size(input_tensor_raw, predictor.plans_manager)

        for step in range(args.steps):
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = net(input_tensor)
                if isinstance(output, (list, tuple)): output = output[0]
                prob_map = torch.softmax(output, dim=1)
                
                # Pseudo-label filtering
                max_probs, pred_classes = torch.max(prob_map, dim=1)
                valid_mask = max_probs > args.conf_thresh
                
                # Spatial Context filtering (Tumor must be near Rectum)
                if num_classes > 2: # Check if tumor class exists
                    rectum_mask = (pred_classes == 1).unsqueeze(1) 
                    rectum_roi = dilate_mask(rectum_mask, kernel_size=ANATOMY_KERNEL_SIZE).squeeze(1)
                    is_tumor = (pred_classes == 2)
                    tumor_violation = is_tumor & (~rectum_roi)
                    valid_mask[tumor_violation] = False 
                
                valid_indices = torch.nonzero(valid_mask.squeeze(0))
                
                if valid_indices.shape[0] > 100000:
                    perm = torch.randperm(valid_indices.shape[0])[:100000]
                    valid_indices = valid_indices[perm]
                    
                if valid_indices.shape[0] < MIN_PIXELS_PER_CLASS:
                    scaler.step(optimizer); scaler.update(); continue

                labels_selected = pred_classes[0, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
                
                total_proto_loss = 0.0
                valid_layer_count = 0
                
                # Feature-Level Proto Loss
                for layer_idx in TARGET_LAYER_IDX_LIST:
                    layer_feat = captured_features_dict[f"layer_{layer_idx}"]
                    target_shape_tensor = torch.tensor(prob_map.shape[2:], device=DEVICE, dtype=torch.float)
                    
                    features_selected = get_point_features(layer_feat, valid_indices, target_shape_tensor)
                    layer_loss = 0.0
                    valid_cls_count = 0
                    
                    present_classes = torch.unique(labels_selected)
                    for cls_id in present_classes:
                        cls_id = cls_id.item()
                        cls_mask = (labels_selected == cls_id)
                        cls_feats = features_selected[cls_mask]
                        if cls_feats.shape[0] == 0: continue
                        
                        cls_feats = F.normalize(cls_feats, dim=1)
                        protos = multi_level_prototypes[layer_idx][cls_id]
                        protos = F.normalize(protos, dim=1)
                        
                        sim_matrix = torch.matmul(cls_feats, protos.t())
                        max_sim, _ = torch.max(sim_matrix, dim=1)
                        layer_loss += (1 - max_sim.mean())
                        valid_cls_count += 1
                    
                    if valid_cls_count > 0:
                        total_proto_loss += LAYER_WEIGHTS[layer_idx] * (layer_loss / valid_cls_count)
                        valid_layer_count += 1
                
                # Output-Level KL Prior Loss
                prob_mean = prob_map.mean(dim=(2, 3, 4))
                loss_prior = F.kl_div(torch.log(prob_mean + 1e-10), target_prior_dist, reduction='batchmean')

                if valid_layer_count == 0:
                    loss = args.lambda_prior * loss_prior
                else:
                    loss = total_proto_loss / valid_layer_count + args.lambda_prior * loss_prior

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            del output, prob_map, features_selected
            torch.cuda.empty_cache()
            
            if step % 5 == 0:
                print(f"   Step {step}: Total Loss={loss.item():.4f}")

        # --- Final Inference & Post-processing ---
        with torch.no_grad():
            net.eval() # Ensure eval mode for final prediction
            final_output = net(input_tensor)
            if isinstance(final_output, (list, tuple)): final_output = final_output[0]
            final_pred_padded = torch.argmax(final_output, dim=1)
            
            od, oh, ow = original_shape
            final_pred = final_pred_padded[0, :od, :oh, :ow]

            if 'transpose_forward' in data_properties:
                inverse_order = np.argsort(data_properties['transpose_forward']).tolist()
                final_pred = final_pred.permute(*inverse_order)
            
            final_arr = final_pred.cpu().numpy().astype(np.uint8)

            # Undo Resampling
            if 'shape_after_cropping_and_before_resampling' in data_properties:
                target_shape_after_crop = data_properties['shape_after_cropping_and_before_resampling']
                if list(final_arr.shape) != list(target_shape_after_crop):
                    final_tensor = torch.from_numpy(final_arr).unsqueeze(0).unsqueeze(0).float()
                    final_restored = F.interpolate(final_tensor, size=target_shape_after_crop, mode='nearest')
                    final_arr = final_restored.squeeze().cpu().numpy().astype(np.uint8)

            # Undo Cropping
            if 'bbox_used_for_cropping' in data_properties:
                bbox = data_properties['bbox_used_for_cropping']
                shape_original = data_properties['shape_before_cropping']
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

        final_img = sitk.GetImageFromArray(final_arr)
        final_img.CopyInformation(img_sitk)
        sitk.WriteImage(final_img, save_path)
        print(f"✅ Saved adapted prediction: {save_path}")

    for handle in hook_handles:
        handle.remove()
    print(f"\n🎉 Adaptation for Fold {CURRENT_FOLD} completed!")