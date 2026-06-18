import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import random
from sklearn.cluster import KMeans
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Tri-Level Source Priors for Tri-KA TTA")
    
    # --- Data & Model Paths (Replace defaults with your own when testing locally) ---
    parser.add_argument('--base_nnunet_dir', type=str, default="/path/to/your/nnUNet/DATASET",
                        help="Base directory for nnUNet containing raw, preprocessed, and results folders.")
    parser.add_argument('--model_folder', type=str, default="DatasetXXX_YourTask/nnUNetTrainer__nnUNetPlans__3d_fullres",
                        help="Relative path inside nnUNet_trained_models to your specific trained model.")
    parser.add_argument('--train_img_dir', type=str, default="/path/to/your/Task/imagesTr",
                        help="Path to source domain training images.")
    parser.add_argument('--train_label_dir', type=str, default="/path/to/your/Task/labelsTr",
                        help="Path to source domain training labels.")
    parser.add_argument('--save_dir', type=str, default="./source_priors",
                        help="Directory to save the extracted priors.")
    
    # --- Task-Specific Parameters (Crucial for adaptation to other datasets) ---
    parser.add_argument('--num_classes', type=int, default=3, 
                        help="Number of classes including background (e.g., 3 for Rectal Cancer: BG, Rectum, Tumor).")
    parser.add_argument('--canonical_h', type=int, default=320, 
                        help="Canonical height for FFT style extraction.")
    parser.add_argument('--canonical_w', type=int, default=320, 
                        help="Canonical width for FFT style extraction.")
    
    # --- TTA Hyperparameters ---
    parser.add_argument('--num_prototypes', type=int, default=5, 
                        help="Number of prototypes per class per layer.")
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4], 
                        help="Which folds to process (e.g., 0 1 2 3 4).")
    
    return parser.parse_args()

args = parse_args()

# ================= 1. Configuration & Environment Setup =================
# Set nnUNet environment variables dynamically based on args
os.environ['nnUNet_raw'] = os.path.join(args.base_nnunet_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(args.base_nnunet_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(args.base_nnunet_dir, 'nnUNet_trained_models')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

NNUNET_MODEL_FOLDER = os.path.join(os.environ['nnUNet_results'], args.model_folder)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

OUTPUT_PRIOR_PATH = os.path.join(args.save_dir, "source_class_prior.npy")
OUTPUT_STYLE_PATH = os.path.join(args.save_dir, "source_style_2d.npy")
OUTPUT_PROTO_PREFIX = os.path.join(args.save_dir, "source_multi_level_prototypes")

# For typical 3D nnU-Net, stages -1 to -7 represent the encoder hierarchy
TARGET_LAYER_IDX_LIST = [-7, -6, -5, -4, -3, -2, -1] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CASE_PER_CLASS = 3000 
MAX_GLOBAL_SAMPLES = 200000 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# ================= 2. Utility Functions =================
def get_network_divisors(net):
    if hasattr(net, 'encoder') and hasattr(net.encoder, 'strides'):
        strides = net.encoder.strides
    else:
        return (64, 64, 64)
    div_d, div_h, div_w = 1, 1, 1
    for s in strides:
        div_d *= s[0]; div_h *= s[1]; div_w *= s[2]
    return (div_d, div_h, div_w)

def pad_to_compatible_size(input_tensor, net):
    div_d, div_h, div_w = get_network_divisors(net)
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

def get_robust_mask(vol):
    thresh = np.percentile(vol, 50) 
    return vol > thresh

def calculate_class_prior(label_dir, num_classes):
    print(f"\n📂 Reading labels from: {label_dir}")
    files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])
    if len(files) == 0:
        raise ValueError("❌ No .nii.gz files found in the directory. Please check the path.")
        
    total_counts = np.zeros(num_classes, dtype=np.int64)
    for filename in tqdm(files, desc="Calculating Class Prior"):
        filepath = os.path.join(label_dir, filename)
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
        counts = np.bincount(label_arr.flatten(), minlength=num_classes)
        if len(counts) > num_classes:
            total_counts += counts[:num_classes]
        else:
            total_counts += counts

    total_pixels = np.sum(total_counts)
    class_prior = total_counts / total_pixels
    return class_prior, total_counts

# ================= 3. Main Pipeline =================

if __name__ == "__main__":
    
    # --- [Task 0: Output-Level Class Prior Extraction] ---
    if not os.path.exists(OUTPUT_PRIOR_PATH):
        print(f"\n{'='*20} Task 0: Class Prior Extraction {'='*20}")
        prior, counts = calculate_class_prior(args.train_label_dir, args.num_classes)
        np.save(OUTPUT_PRIOR_PATH, prior)
        print(f"✅ Saved Class Prior to {OUTPUT_PRIOR_PATH}")
        print("🚀 Class Prior Distribution:")
        for i, p in enumerate(prior):
            print(f"  Class {i}: {p*100:.4f}%")
    else:
        print(f"\n✅ Task 0: Class Prior already extracted at {OUTPUT_PRIOR_PATH}. Skipping...")

    # Get all training files
    train_files = sorted([f for f in os.listdir(args.train_img_dir) if f.endswith(".nii.gz")])

    # Initialize Predictor
    print(f"\n{'='*20} Initializing Predictor Wrapper {'='*20}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=DEVICE, verbose=False, allow_tqdm=True
    )

    is_style_extracted = os.path.exists(OUTPUT_STYLE_PATH)
    if is_style_extracted:
        print(f"✅ Task 1: Style Prior already exists at {OUTPUT_STYLE_PATH}. Will skip calculation.")
    style_accumulators = [] 

    # === Fold Iteration ===
    for fold in args.folds:
        print(f"\n{'='*20} Processing FOLD {fold} {'='*20}")
        
        # 1. Load weights for specific fold
        try:
            predictor.initialize_from_trained_model_folder(
                NNUNET_MODEL_FOLDER, 
                use_folds=(fold,), 
                checkpoint_name='checkpoint_best.pth'
            )
        except Exception as e:
            print(f"❌ Error loading Fold {fold}: {e}")
            print("Skipping this fold...")
            continue
        
        preprocessor_class = predictor.configuration_manager.preprocessor_class
        predictor.preprocessor = preprocessor_class(verbose=predictor.verbose_preprocessing)
        net = predictor.network
        net.to(DEVICE)
        net.eval()
        
        # 2. Register Hooks for deep layers
        captured_features_dict = {}
        def make_hook(layer_name):
            def hook(module, input, output):
                captured_features_dict[layer_name] = output
            return hook

        hook_handles = []
        for idx in TARGET_LAYER_IDX_LIST:
            layer_name = f"layer_{idx}"
            target_module = net.encoder.stages[idx]
            handle = target_module.register_forward_hook(make_hook(layer_name))
            hook_handles.append(handle)
            
        # 3. Iterate over data
        multi_level_features = {idx: {c: [] for c in range(args.num_classes)} for idx in TARGET_LAYER_IDX_LIST}
        
        print(f"Extracting features for Fold {fold}...")
        with torch.no_grad():
            for idx, case_file in enumerate(train_files):
                print(f"Fold {fold} | Case {idx+1}/{len(train_files)}: {case_file}", end='\r')
                
                img_path = os.path.join(args.train_img_dir, case_file)
                lbl_path = os.path.join(args.train_label_dir, case_file.split('.')[0][:-5] + '.nii.gz')
                
                data, seg, properties = predictor.preprocessor.run_case(
                    [img_path], lbl_path, predictor.plans_manager, 
                    predictor.configuration_manager, predictor.dataset_json
                )
                img_np = data[0]
                
                # --- [Task 1: Input-Level Style Extraction] ---
                if not is_style_extracted:
                    try:
                        slice_indices = np.linspace(0, img_np.shape[0]-1, 5).astype(int)
                        case_amp_sum = np.zeros((args.canonical_h, args.canonical_w), dtype=np.float32)
                        
                        for slice_idx in slice_indices:
                            slice_img = img_np[slice_idx]
                            fft = np.fft.fft2(slice_img)
                            amp = np.abs(np.fft.fftshift(fft))
                            
                            if amp.shape != (args.canonical_h, args.canonical_w):
                                amp_t = torch.from_numpy(amp).unsqueeze(0).unsqueeze(0)
                                amp = F.interpolate(amp_t, size=(args.canonical_h, args.canonical_w), mode='bilinear', align_corners=False).squeeze().numpy()
                            case_amp_sum += amp
                        style_accumulators.append(case_amp_sum / len(slice_indices))
                    except Exception as e:
                        pass

                # --- [Task 2: Feature-Level Prototype Extraction] ---
                input_tensor_raw = torch.from_numpy(data).to(DEVICE).unsqueeze(0)
                seg_tensor_raw = torch.from_numpy(seg).to(DEVICE).unsqueeze(0)
                
                input_tensor, pad_shape = pad_to_compatible_size(input_tensor_raw, net)
                min_lbl = seg_tensor_raw.min()
                d, h, w = seg_tensor_raw.shape[2:]
                pd, ph, pw = pad_shape[0]-d, pad_shape[1]-h, pad_shape[2]-w
                seg_tensor = F.pad(seg_tensor_raw, (0, pw, 0, ph, 0, pd), mode='constant', value=min_lbl)

                _ = net(input_tensor)
                
                for layer_idx in TARGET_LAYER_IDX_LIST:
                    layer_name = f"layer_{layer_idx}"
                    features = captured_features_dict[layer_name]
                    seg_resized = F.interpolate(seg_tensor.float(), size=features.shape[2:], mode='nearest')
                    feat_dim = features.shape[1]
                    z_flat = features.squeeze(0).view(feat_dim, -1).permute(1, 0).cpu().numpy()
                    seg_flat = seg_resized.flatten().cpu().numpy()
                    
                    for cls_id in range(args.num_classes):
                        mask = (seg_flat == cls_id)
                        if not np.any(mask): continue
                        cls_features = z_flat[mask]
                        if cls_features.shape[0] > SAMPLES_PER_CASE_PER_CLASS:
                            indices = np.random.choice(cls_features.shape[0], SAMPLES_PER_CASE_PER_CLASS, replace=False)
                            cls_features_sampled = cls_features[indices]
                        else:
                            cls_features_sampled = cls_features
                        multi_level_features[layer_idx][cls_id].append(cls_features_sampled)

        # 4. Save Style Prior
        if not is_style_extracted and len(style_accumulators) > 0:
            print(f"\n\n--> Saving Common Style Prior to {OUTPUT_STYLE_PATH}...")
            all_amps_np = np.array(style_accumulators)
            avg_amp_global = np.mean(all_amps_np, axis=0) 
            np.save(OUTPUT_STYLE_PATH, avg_amp_global)
            is_style_extracted = True 

        # 5. Cluster and Save Prototypes
        current_proto_path = f"{OUTPUT_PROTO_PREFIX}_fold{fold}.npz"
        print(f"--> Clustering Prototypes for FOLD {fold}...")
        
        multi_level_prototypes = {} 
        for layer_idx in TARGET_LAYER_IDX_LIST:
            multi_level_prototypes[layer_idx] = {}
            for cls_id in range(args.num_classes):
                features_list = multi_level_features[layer_idx][cls_id]
                if not features_list:
                    feat_dim_layer = multi_level_features[layer_idx][0][0].shape[1]
                    multi_level_prototypes[layer_idx][cls_id] = np.zeros((args.num_prototypes, feat_dim_layer))
                    continue
                    
                all_feats = np.concatenate(features_list, axis=0)
                if all_feats.shape[0] > MAX_GLOBAL_SAMPLES:
                    indices = np.random.choice(all_feats.shape[0], MAX_GLOBAL_SAMPLES, replace=False)
                    train_feats = all_feats[indices]
                else:
                    train_feats = all_feats
                    
                kmeans = KMeans(n_clusters=args.num_prototypes, random_state=42, n_init=10) 
                kmeans.fit(train_feats)
                prototypes = kmeans.cluster_centers_
                prototypes = F.normalize(torch.from_numpy(prototypes), dim=1, p=2).numpy()
                multi_level_prototypes[layer_idx][cls_id] = prototypes

        save_dict = {}
        for layer_idx in TARGET_LAYER_IDX_LIST:
            for cls_id in range(args.num_classes):
                key = f"layer_{layer_idx}_class_{cls_id}"
                save_dict[key] = multi_level_prototypes[layer_idx][cls_id]
            sample_proto = multi_level_prototypes[layer_idx][0]
            save_dict[f"layer_{layer_idx}_feat_dim"] = np.array([sample_proto.shape[1]])

        np.savez(current_proto_path, **save_dict)
        print(f"✅ Saved Feature Prototypes: {current_proto_path}")
        
        for handle in hook_handles:
            handle.remove()
        
        del net
        torch.cuda.empty_cache()

    print(f"\n{'='*40}")
    print("🎉 All Source Priors (Class Distribution, Style, and Prototypes) Extracted Successfully!")
    print(f"📁 Check '{args.save_dir}' for output files.")
    print(f"{'='*40}")