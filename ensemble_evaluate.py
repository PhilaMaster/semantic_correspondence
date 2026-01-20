import json
from collections import defaultdict
import numpy as np
from sklearn import base
import torch
import torch.nn.functional as F
import time
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

from SPair71k.devkit.SPairDataset import SPairDataset
from pf_pascal.PFPascalDataset import PFPascalDataset
from helper_functions import extract_dense_features, extract_dense_features_SAM, pixel_to_patch_coord, patch_to_pixel_coord
from matching_strategies import find_best_match_argmax, find_best_match_window_softargmax
from pck import compute_pck_spair71k, compute_pck_pfpascal
from models.dinov3.dinov3.models.vision_transformer import vit_base as dinov3_vit_base
from models.dinov2.dinov2.models.vision_transformer import vit_base as dinov2_vit_base
from models.segment_anything.segment_anything import sam_model_registry

# ==================== CONFIG ====================
IMG_SIZE_DINOV2 = 518
PATCH_SIZE_DINOV2 = 14
IMG_SIZE_DINOV3 = 512
PATCH_SIZE_DINOV3 = 16
IMG_SIZE_SAM = 512
PATCH_SIZE_SAM = 16

BASE = 'SPair71k'
PAIR_ANN_PATH = f'{BASE}/PairAnnotation'
LAYOUT_PATH = f'{BASE}/Layout'
IMAGE_PATH = f'{BASE}/JPEGImages'
DATASET_SIZE = 'large'
PCK_ALPHA = 0.1
THRESHOLDS = [0.05, 0.1, 0.2]

# Feature fusion strategies
FUSION_STRATEGIES = {
    'concatenation': 'concat',  # Concatenate features [D1+D2+D3]
    'average': 'avg',            # Average features
    'weighted_average': 'weighted_avg'  # Weighted average
}

CHECKPOINT_PATHS = {
    "DINOv2": "models/dinov2/weights/dinov2_vitb14_finetuned_only_model_10temp.pth",
    "DINOv3": "models/dinov3/weights/finetuned/dinov3_vitb16_finetuned_3bl_0.0001lr_15t.pth",
    "SAM": "models/segment_anything/weights/finetuned/SAM_finetuned_4bl_15t_0.0001lr.pth"
}


# ==================== HELPER FUNCTIONS ====================

def load_models(device):
    """Load all three finetuned models."""
    print("Loading models...")
    
    # DINOv2
    dinov2 = dinov2_vit_base(
        img_size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2),
        patch_size=PATCH_SIZE_DINOV2,
        num_register_tokens=0,
        block_chunks=0,
        init_values=1.0,
    )
    ckpt_dinov2 = torch.load(CHECKPOINT_PATHS["DINOv2"], map_location=device)
    dinov2.load_state_dict(ckpt_dinov2, strict=True)
    dinov2.to(device)
    dinov2.eval()
    
    # DINOv3
    dinov3 = dinov3_vit_base(
        img_size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3),
        patch_size=PATCH_SIZE_DINOV3,
        n_storage_tokens=4,
        layerscale_init=1.0,
        mask_k_bias=True,
    )
    ckpt_dinov3 = torch.load(CHECKPOINT_PATHS["DINOv3"], map_location=device)
    dinov3.load_state_dict(ckpt_dinov3["model_state_dict"], strict=True)
    dinov3.to(device)
    dinov3.eval()
    
    # SAM
    # Initialize the SAM model without loading checkpoint yet
    sam = sam_model_registry["vit_b"](checkpoint=None) # Pass None to initialize without loading
    sam.to(device)

    # Load the custom finetuned checkpoint
    print(f"Loading finetuned SAM checkpoint from {CHECKPOINT_PATHS['SAM']}")
    checkpoint = torch.load(CHECKPOINT_PATHS["SAM"], map_location=device)

    # The finetuned checkpoint likely contains more than just the model state_dict.
    # Extract the actual model_state_dict and load it
    if 'model_state_dict' in checkpoint:
        sam.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded 'model_state_dict' from checkpoint.")
    else:
        # If the checkpoint itself is just the state_dict, try loading it directly
        sam.load_state_dict(checkpoint)
        print("Successfully loaded checkpoint directly as state_dict.")
    sam.eval()
    
    print("✓ All models loaded successfully")
    return dinov2, dinov3, sam


def normalize_features(features):
    """L2 normalize features along the feature dimension."""
    # features: [B, H, W, D] or [H*W, D]
    if len(features.shape) == 4:
        # [B, H, W, D] -> normalize over D
        return F.normalize(features, p=2, dim=-1)
    else:
        # [H*W, D] -> normalize over D
        return F.normalize(features, p=2, dim=1)


def fuse_features(src_features_list, fusion_strategy='avg', weights=None):
    """
    Fuse features from multiple models.
    
    Args:
        src_features_list: list of [D] tensors
        fusion_strategy: 'avg', 'concat', or 'weighted_avg'
        weights: for 'weighted_avg', list of weights (should sum to 1)
    
    Returns:
        fused_feature: [D_fused] tensor
    """
    if fusion_strategy == 'avg':
        # Simple average
        fused = torch.stack(src_features_list, dim=0).mean(dim=0)
        
    elif fusion_strategy == 'concat':
        # Concatenate all features
        fused = torch.cat(src_features_list, dim=0)
        
    elif fusion_strategy == 'weighted_avg':
        if weights is None:
            weights = [1/len(src_features_list)] * len(src_features_list)
        # Weighted average
        fused = sum(w * f for w, f in zip(weights, src_features_list))
        
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    return fused


def fuse_target_features_batch(tgt_features_list, target_shape, fusion_strategy='avg', weights=None):
    """
    Fuse target feature maps.
    
    Args:
        tgt_features_list: list of [H, W, D] tensors (after removing batch dimension)
        target_shape: shape of largest feature map to use as reference
        fusion_strategy: 'avg', 'concat', or 'weighted_avg'
        weights: for 'weighted_avg'
    
    Returns:
        fused_features_flat: [H*W, D_fused] tensor
    """
    # If features have different spatial dimensions, interpolate to largest
    aligned_features = []
    for feat in tgt_features_list:
        if feat.shape[:2] != target_shape[:2]:
            # Reshape to [1, D, H, W] for interpolation
            feat_reshaped = feat.permute(2, 0, 1).unsqueeze(0)
            feat_reshaped = F.interpolate(
                feat_reshaped,
                size=target_shape[:2],
                mode='bilinear',
                align_corners=False
            )
            feat = feat_reshaped.squeeze(0).permute(1, 2, 0)
        aligned_features.append(feat)
    
    # Fuse features
    if fusion_strategy == 'avg':
        fused = torch.stack(aligned_features, dim=0).mean(dim=0)
    elif fusion_strategy == 'concat':
        fused = torch.cat(aligned_features, dim=-1)
    elif fusion_strategy == 'weighted_avg':
        if weights is None:
            weights = [1/len(aligned_features)] * len(aligned_features)
        fused = sum(w * f for w, f in zip(weights, aligned_features))
    
    # Flatten: [H, W, D] -> [H*W, D]
    H, W, D = fused.shape
    fused_flat = fused.reshape(H * W, D)
    
    return fused_flat


def evaluate_ensemble_with_params(
    models_dict,
    dataset,
    device,
    K,
    temperature,
    fusion_strategy='avg',
    weights=None,
    thresholds=None
):
    """
    Evaluate ensemble with specific parameters.
    
    Args:
        models_dict: dict with 'dinov2', 'dinov3', 'sam' models
        dataset: evaluation dataset
        device: torch device
        K: window size for softargmax
        temperature: softmax temperature
        fusion_strategy: 'avg', 'concat', or 'weighted_avg'
        weights: [w_dinov2, w_dinov3, w_sam] for weighted fusion
        thresholds: PCK thresholds
    
    Returns:
        per_image_metrics: list of dicts with PCK scores
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    
    per_image_metrics = []
    dinov2, dinov3, sam = models_dict['dinov2'], models_dict['dinov3'], models_dict['sam']
    
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            # Load and resize images
            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)
            
            # Resize for DINOv2
            src_dinov2 = F.interpolate(src_tensor, size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2), 
                                       mode='bilinear', align_corners=False)
            tgt_dinov2 = F.interpolate(tgt_tensor, size=(IMG_SIZE_DINOV2, IMG_SIZE_DINOV2), 
                                       mode='bilinear', align_corners=False)
            
            # Resize for DINOv3
            src_dinov3 = F.interpolate(src_tensor, size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3), 
                                       mode='bilinear', align_corners=False)
            tgt_dinov3 = F.interpolate(tgt_tensor, size=(IMG_SIZE_DINOV3, IMG_SIZE_DINOV3), 
                                       mode='bilinear', align_corners=False)
            
            # Resize for SAM
            src_sam = F.interpolate(src_tensor, size=(IMG_SIZE_SAM, IMG_SIZE_SAM), 
                                    mode='bilinear', align_corners=False)
            tgt_sam = F.interpolate(tgt_tensor, size=(IMG_SIZE_SAM, IMG_SIZE_SAM), 
                                    mode='bilinear', align_corners=False)
            
            # Extract features from all models
            src_feat_dinov2 = extract_dense_features(dinov2, src_dinov2)
            tgt_feat_dinov2 = extract_dense_features(dinov2, tgt_dinov2)
            
            src_feat_dinov3 = extract_dense_features(dinov3, src_dinov3)
            tgt_feat_dinov3 = extract_dense_features(dinov3, tgt_dinov3)
            
            src_feat_sam = extract_dense_features_SAM(sam, src_sam, image_size=IMG_SIZE_SAM)
            tgt_feat_sam = extract_dense_features_SAM(sam, tgt_sam, image_size=IMG_SIZE_SAM)
            
            # Get original sizes
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])
            
            # Get keypoints
            src_kps = sample['src_kps'].numpy()
            trg_kps = sample['trg_kps'].numpy()
            kps_ids = sample['kps_ids']
            # trg_bbox = sample['trg_bbox']
            category = sample['category']
            
            # Prepare target features for batch processing
            tgt_feat_dinov2_squeezed = tgt_feat_dinov2.squeeze(0)  # [H2, W2, D2]
            tgt_feat_dinov3_squeezed = tgt_feat_dinov3.squeeze(0)  # [H3, W3, D3]
            tgt_feat_sam_squeezed    = tgt_feat_sam.squeeze(0)     # [Hs, Ws, Ds]

            # Use SAM grid as the reference grid
            ref_shape = tgt_feat_sam_squeezed.shape  # (Hs, Ws, Ds)
            H_ref, W_ref = ref_shape[0], ref_shape[1]

            # Precompute normalized target flats for score-level fusion
            H2, W2, D2 = tgt_feat_dinov2_squeezed.shape
            H3, W3, D3 = tgt_feat_dinov3_squeezed.shape
            Hs, Ws, Ds = tgt_feat_sam_squeezed.shape

            tgt_v2_flat = F.normalize(tgt_feat_dinov2_squeezed.reshape(H2 * W2, D2), dim=1)
            tgt_v3_flat = F.normalize(tgt_feat_dinov3_squeezed.reshape(H3 * W3, D3), dim=1)
            tgt_s_flat  = F.normalize(tgt_feat_sam_squeezed.reshape(Hs * Ws, Ds),    dim=1)

            # Only build a fused feature map for feature-level fusion
            if fusion_strategy in ('avg', 'concat'):
                tgt_fused_flat = fuse_target_features_batch(
                    [tgt_feat_dinov2_squeezed, tgt_feat_dinov3_squeezed, tgt_feat_sam_squeezed],
                    ref_shape,
                    fusion_strategy=fusion_strategy,
                    weights=weights
                )
                tgt_fused_flat = normalize_features(tgt_fused_flat)  # [H_ref*W_ref, D_fused]
            else:
                tgt_fused_flat = None

            pred_matches = []
            
            # Process each keypoint
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]

                # Source features per model
                px2, py2 = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                patch_size=PATCH_SIZE_DINOV2, resized_size=IMG_SIZE_DINOV2)
                src_v2 = F.normalize(src_feat_dinov2[0, py2, px2, :], dim=0)

                px3, py3 = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                patch_size=PATCH_SIZE_DINOV3, resized_size=IMG_SIZE_DINOV3)
                src_v3 = F.normalize(src_feat_dinov3[0, py3, px3, :], dim=0)

                pxs, pys = pixel_to_patch_coord(src_x, src_y, src_original_size,
                                                patch_size=PATCH_SIZE_SAM, resized_size=IMG_SIZE_SAM)
                src_vs = F.normalize(src_feat_sam[0, pys, pxs, :], dim=0)

                if fusion_strategy == 'weighted_avg':
                    # Score-level fusion: build per-model sim maps, upsample to ref grid, then weight-sum
                    sim2 = F.cosine_similarity(src_v2.unsqueeze(0), tgt_v2_flat, dim=1).view(H2, W2)
                    sim3 = F.cosine_similarity(src_v3.unsqueeze(0), tgt_v3_flat, dim=1).view(H3, W3)
                    sims = F.cosine_similarity(src_vs.unsqueeze(0),  tgt_s_flat,  dim=1).view(Hs, Ws)

                    def resize_map(m, H_t, W_t):
                        if (H_t, W_t) == (H_ref, W_ref):
                            return m
                        return F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H_ref, W_ref),
                                             mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                    sim2_r = resize_map(sim2, H2, W2)
                    sim3_r = resize_map(sim3, H3, W3)
                    sims_r = resize_map(sims, Hs, Ws)

                    similarities = (weights[0] * sim2_r + weights[1] * sim3_r + weights[2] * sims_r).reshape(-1)
                    width_eff, height_eff = W_ref, H_ref
                else:
                    # Feature-level fusion (avg/concat) as before
                    src_fused = fuse_features([src_v2, src_v3, src_vs],
                                              fusion_strategy=fusion_strategy,
                                              weights=weights)
                    src_fused = F.normalize(src_fused.unsqueeze(0), dim=1).squeeze(0)
                    similarities = F.cosine_similarity(src_fused.unsqueeze(0), tgt_fused_flat, dim=1)
                    width_eff, height_eff = W_ref, H_ref  # fused grid equals ref grid

                # Find best match on ensemble similarity map
                match_patch_x, match_patch_y = find_best_match_window_softargmax(
                    similarities, width_eff, height_eff, K=K, temperature=temperature
                )

                # Convert to original image coords (ref grid = SAM)
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size,
                    patch_size=PATCH_SIZE_SAM, resized_size=IMG_SIZE_SAM
                )
                pred_matches.append([match_x, match_y])
            
            # Compute PCK
            image_pcks = {}
            for threshold in thresholds:
                # pck, _, _ = compute_pck_spair71k(
                #     pred_matches,
                #     trg_kps.tolist(),
                #     trg_bbox,
                #     threshold
                # )
                pck, correct_mask, distances = compute_pck_pfpascal(
                    pred_matches, trg_kps, tgt_original_size, threshold
                    )
                image_pcks[threshold] = pck
            per_image_metrics.append({
                'category': category,
                'pck_scores': image_pcks,
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} images")
    
    return per_image_metrics


def run_grid_search_ensemble(models_dict, val_dataset, device, results_dir):
    """Run grid search over ensemble parameters."""
    
    # Hyperparameter ranges
    K_values = [3, 5, 7]
    temperature_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    fusion_strategies = ['avg', 'concat', 'weighted_avg']
    weight_combinations = [
        [1/3, 1/3, 1/3],           # Equal
        [0.4, 0.3, 0.3],            # DINOv2 priority
        [0.3, 0.4, 0.3],            # DINOv3 priority
        [0.3, 0.3, 0.4],            # SAM priority
    ]
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GRID SEARCH FOR ENSEMBLE PARAMETERS")
    print("=" * 80)
    print(f"K values: {K_values}")
    print(f"Temperature values: {temperature_values}")
    print(f"Fusion strategies: {fusion_strategies}")
    print(f"Weight combinations: {len(weight_combinations)}")
    print(f"Total combinations: {len(K_values) * len(temperature_values) * len(fusion_strategies) * len(weight_combinations)}")
    print(f"Validation set size: {len(val_dataset)}")
    print("=" * 80)
    
    all_results = []
    total_combinations = len(K_values) * len(temperature_values) * len(fusion_strategies) * len(weight_combinations)
    current_combo = 0
    
    for fusion_strategy in fusion_strategies:
        for weights in weight_combinations if fusion_strategy == 'weighted_avg' else [None]:
            for K in K_values:
                for temp in temperature_values:
                    current_combo += 1
                    weight_str = f"w={weights}" if weights else "default"
                    print(f"\n[{current_combo}/{total_combinations}] "
                          f"Fusion={fusion_strategy}, {weight_str}, K={K}, temp={temp}")
                    
                    start_time = time.time()
                    per_image_metrics = evaluate_ensemble_with_params(
                        models_dict, val_dataset, device, K, temp,
                        fusion_strategy=fusion_strategy,
                        weights=weights
                    )
                    inference_time = time.time() - start_time
                    
                    result = {
                        'fusion_strategy': fusion_strategy,
                        'weights': str(weights),
                        'K': K,
                        'temperature': temp,
                        'inference_time_sec': inference_time,
                    }
                    
                    for threshold in THRESHOLDS:
                        all_pcks = np.array([img['pck_scores'][threshold] for img in per_image_metrics])
                        result[f'pck@{threshold:.2f}_mean'] = float(np.mean(all_pcks))
                        result[f'pck@{threshold:.2f}_std'] = float(np.std(all_pcks))
                        result[f'pck@{threshold:.2f}_median'] = float(np.median(all_pcks))
                    
                    all_results.append(result)
                    
                    # Save intermediate results
                    df_results = pd.DataFrame(all_results)
                    df_results.to_csv(f'{results_dir}/grid_search_results.csv', index=False)
    
    # Find best configuration
    best_idx = np.argmax([r['pck@0.10_mean'] for r in all_results])
    best_result = all_results[best_idx]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION FOUND:")
    print("=" * 80)
    print(json.dumps(best_result, indent=2))
    
    with open(f'{results_dir}/best_config.json', 'w') as f:
        json.dump(best_result, f, indent=2)
    
    return all_results




# ==================== MAIN ====================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    dinov2, dinov3, sam = load_models(device)
    models_dict = {'dinov2': dinov2, 'dinov3': dinov3, 'sam': sam}

    # Fixed window soft-argmax params
    K = 5
    temperature = 0.2

    # Weighted-average fusion weights: [DINOv2, DINOv3, SAM]
    # Start equal; adjust based on validation if desired (e.g., [0.4, 0.3, 0.3])
    weights = [0.35, 0.45, 0.20]

    # Load test dataset
    print("\nLoading test dataset...")
    # test_dataset = SPairDataset(PAIR_ANN_PATH, LAYOUT_PATH, IMAGE_PATH, DATASET_SIZE, 
    #                             PCK_ALPHA, datatype='test')
    base = 'pf_pascal'
    test_dataset = PFPascalDataset(base, split='test')
    print(f"✓ Test set loaded: {len(test_dataset)} pairs")

    # Results dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wtag = f"{weights[0]:.2f}-{weights[1]:.2f}-{weights[2]:.2f}"
    results_dir = f'results_SPair71K/ensemble/weighted_avg/K{K}_T{temperature}_w{wtag}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Evaluate with weighted_avg fusion
    start = time.time()
    per_image_metrics = evaluate_ensemble_with_params(
        models_dict=models_dict,
        dataset=test_dataset,
        device=device,
        K=K,
        temperature=temperature,
        fusion_strategy='weighted_avg',
        weights=weights,
        thresholds=THRESHOLDS
    )
    elapsed = time.time() - start
    print(f"Total inference time: {elapsed:.2f} seconds")

    # Aggregate overall stats
    overall_stats = {"inference_time_sec": elapsed}
    for threshold in THRESHOLDS:
        all_pcks = np.array([img['pck_scores'][threshold] for img in per_image_metrics])
        overall_stats[f"pck@{threshold:.2f}"] = {
            "mean": float(np.mean(all_pcks)),
            "std": float(np.std(all_pcks)),
            "median": float(np.median(all_pcks)),
            "p25": float(np.percentile(all_pcks, 25)),
            "p75": float(np.percentile(all_pcks, 75)),
        }
        print(f"PCK@{threshold:.2f}: mean={overall_stats[f'pck@{threshold:.2f}']['mean']:.2f}% "
              f"std={overall_stats[f'pck@{threshold:.2f}']['std']:.2f}% "
              f"median={overall_stats[f'pck@{threshold:.2f}']['median']:.2f}% "
              f"p25={overall_stats[f'pck@{threshold:.2f}']['p25']:.2f}% "
              f"p75={overall_stats[f'pck@{threshold:.2f}']['p75']:.2f}%")

    # Save outputs
    with open(f'{results_dir}/overall_stats.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)
    df_all = pd.DataFrame([
        {"category": m["category"], **{f"pck@{t:.2f}": m["pck_scores"][t] for t in THRESHOLDS}}
        for m in per_image_metrics
    ])
    df_all.to_csv(f'{results_dir}/per_image_metrics.csv', index=False)
    print(f"Saved overall_stats.json and per_image_metrics.csv to {results_dir}")
    

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # Load models
#     dinov2, dinov3, sam = load_models(device)
#     models_dict = {'dinov2': dinov2, 'dinov3': dinov3, 'sam': sam}
    
#     # Load validation dataset
#     print(f"\nLoading validation dataset...")
#     val_dataset = SPairDataset(PAIR_ANN_PATH, LAYOUT_PATH, IMAGE_PATH, DATASET_SIZE, 
#                                 PCK_ALPHA, datatype='val')
#     print(f"✓ Validation set loaded: {len(val_dataset)} pairs")
    
#     # Create results directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = f'results_SPair71K/ensemble_fusion_{timestamp}'
#     os.makedirs(results_dir, exist_ok=True)
    
#     # Run grid search
#     print(f"\nResults will be saved to: {results_dir}\n")
#     run_grid_search_ensemble(models_dict, val_dataset, device, results_dir)
    
#     print(f"\n✓ Grid search completed. Results saved to {results_dir}")

