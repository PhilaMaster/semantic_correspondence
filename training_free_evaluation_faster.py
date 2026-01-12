import json
from collections import defaultdict

import numpy as np
import torch
import time

from SPair71k.devkit.SPairDataset import SPairDataset
# from pf_pascal.PFPascalDataset import PFPascalDataset
from helper_functions import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from matching_strategies import find_best_match_argmax, find_best_match_window_softargmax
from pck import compute_pck_spair71k, compute_pck_pfpascal
from models.dinov3.dinov3.models.vision_transformer import vit_base
# from models.dinov2.dinov2.models.vision_transformer import vit_base
import torch.nn.functional as F
import os
from datetime import datetime
import pandas as pd

patch_size = 16
img_size = 512

base = 'SPair71k'  #path to SPair71k dataset
pair_ann_path = f'{base}/PairAnnotation'
layout_path = f'{base}/Layout'
image_path = f'{base}/JPEGImages'
dataset_size = 'large'
pck_alpha = 0.1 #mock, it's not used in evaluation
use_windowed_softargmax = False

#results_SPair71K folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results_SPair71k/dinov3/finetuned/dinov3_base_spair71k'
results_dir+= '_wsoftargmax_' if use_windowed_softargmax else '_argmax_'
results_dir+=timestamp
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")



model = vit_base (
    img_size= (img_size, img_size),        # base / nominal size
    patch_size=patch_size,             # patch size that matches the checkpoint
    n_storage_tokens=4,
    layerscale_init= 1.0,
    mask_k_bias=True,
)

# model = vit_base(
#     img_size=(img_size, img_size),        # base / nominal size
#     patch_size=14,             # patch size that matches the checkpoint
#     num_register_tokens=0,     # <- no registers
#     block_chunks=0,
#     init_values=1.0,  # LayerScale initialization
# )

device = "cuda" if torch.cuda.is_available() else "cpu" #use GPU if available
print("Using device:", device)
ckpt = torch.load("models/dinov3/weights/finetuned/dinov3_vitb16_finetuned_3bl_0.0001lr_15t.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.to(device)
model.eval()

thresholds = [0.05, 0.1, 0.2]
per_image_metrics = []
all_keypoint_metrics = []
category_metrics = defaultdict(lambda: defaultdict(list))


test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='test')
# test_dataset = PFPascalDataset(base, split='test')
inference_start_time = time.time()

for idx, sample in enumerate(test_dataset):  # type: ignore
    #extract tensors and move to device
    src_tensor = sample['src_img'].unsqueeze(0).to(device)  # [1, 3, H, W]
    tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)  # [1, 3, H, W]

    #resize to img_size x img_size
    src_tensor = F.interpolate(src_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
    tgt_tensor = F.interpolate(tgt_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

    #save original sizes
    src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
    tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

    #extract dense features
    src_features = extract_dense_features(model, src_tensor)
    tgt_features = extract_dense_features(model, tgt_tensor)

    #reshape
    _, H, W, D = tgt_features.shape #B=1
    tgt_flat = tgt_features.reshape(H * W, D)

    #extract keypoints
    src_kps = sample['src_kps'].numpy()  # [N, 2]
    trg_kps = sample['trg_kps'].numpy()  # [N, 2]
    kps_ids = sample['kps_ids']          # [N]

    trg_bbox = sample['trg_bbox']

    pred_matches = []

    #iterate over keypoints and predict matches
    for i in range(src_kps.shape[0]):
        src_x, src_y = src_kps[i]
        tgt_x, tgt_y = trg_kps[i]

        patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size=patch_size, resized_size=img_size)

        #extract source feature at the keypoint patch
        src_feature = src_features[0, patch_y, patch_x, :]  # [D]

        #compute cosine similarities with all target features
        similarities = F.cosine_similarity(
            src_feature.unsqueeze(0),  # [1, D]
            tgt_flat,  # [H*W, D]
            dim=1
        )  # [H*W]

        #find best matching patch in target
        
        # match_patch_x, match_patch_y = find_best_match_window_softargmax(similarities, W, H, K=5, temperature=0.1)
        
        match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)
        match_x, match_y = patch_to_pixel_coord(
            match_patch_x, match_patch_y, tgt_original_size, patch_size=patch_size, resized_size=img_size
        )

        pred_matches.append([match_x, match_y])

    #compute PCK per diverse threshold
    image_pcks = {}
    category = sample['category']

    for threshold in thresholds:
        pck, correct_mask, distances = compute_pck_spair71k(
            pred_matches,
            trg_kps.tolist(),
            trg_bbox,  # (W, H)
            threshold
        )
        # pck, correct_mask, distances = compute_pck_pfpascal(
        #     pred_matches, trg_kps, tgt_original_size, threshold
        # )
        image_pcks[threshold] = pck
        category_metrics[category][threshold].append(pck)

        #store keypoint-wise metrics
        for kps_id, pred, gt, dist, correct in zip(
                kps_ids, pred_matches, trg_kps.tolist(), distances, correct_mask
        ):
            all_keypoint_metrics.append({
                'image_idx': idx,
                'category': category,
                'keypoint_id' : kps_id,
                'pred': pred,
                'gt': gt,
                'distance': dist,
                'correct_at_threshold': correct,
                'threshold': threshold
            })

    #store per-image metrics
    per_image_metrics.append({
        'category': category,
        'source_path': str(sample['src_imname']),
        'target_path': str(sample['trg_imname']),
        'num_keypoints': src_kps.shape[0],
        'pck_scores': image_pcks,
        'pred_points': pred_matches,
        'gt_points': trg_kps.tolist(),
        'kps_ids': kps_ids,
    })

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} pairs...")

    #debug early stopping
    #if idx == 100:
        #break

inference_end_time = time.time()
total_inference_time_sec = inference_end_time - inference_start_time
print(f"Total inference time: {total_inference_time_sec:.2f} seconds")

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

overall_stats = {"inference_time_sec":total_inference_time_sec}

for threshold in thresholds:
    all_pcks = np.array([img['pck_scores'][threshold] for img in per_image_metrics])

    mean_pck = float(np.mean(all_pcks))
    std_pck = float(np.std(all_pcks))
    median_pck = float(np.median(all_pcks))
    p25 = float(np.percentile(all_pcks, 25))
    p75 = float(np.percentile(all_pcks, 75))

    overall_stats[f"pck@{threshold:.2f}"] = {
        "mean": mean_pck,
        "std": std_pck,
        "median": median_pck,
        "p25": p25,
        "p75": p75,
    }

    print(f"PCK@{threshold:.2f}: "
          f"mean={mean_pck:.2f}%, std={std_pck:.2f}%, "
          f"median={median_pck:.2f}%, "
          f"p25={p25:.2f}%, p75={p75:.2f}%")

with open(f'{results_dir}/overall_stats.json', 'w') as f:
    json.dump(overall_stats, f, indent=2)

df_all_kp = pd.DataFrame(all_keypoint_metrics)
csv_path = f'{results_dir}/all_keypoint_metrics.csv'
df_all_kp.to_csv(csv_path, index=False)
print(f"Saved all keypoint metrics to '{csv_path}'")
