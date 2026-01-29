import json
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from SPair71k.devkit.SPairDataset import SPairDataset
from helper_functions import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from matching_strategies import find_best_match_window_softargmax
# from models.dinov2.dinov2.models.vision_transformer import vit_base
from models.dinov3.dinov3.models.vision_transformer import vit_base
from pck import compute_pck_spair71k



def evaluate_with_params(model, dataset, device, K, temperature, img_size, patch_size, thresholds=[0.05, 0.1, 0.2]):
    """Evaluate model with specific K and temperature parameters."""
    per_image_metrics = []

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)

            # resize to 518x518
            src_tensor = F.interpolate(src_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

            # save original sizes
            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # extract dense features
            src_features = extract_dense_features(model, src_tensor)
            tgt_features = extract_dense_features(model, tgt_tensor)

            # reshape
            _, H, W, D = tgt_features.shape
            tgt_flat = tgt_features.reshape(H * W, D)

            # extract keypoints
            src_kps = sample['src_kps'].numpy()
            trg_kps = sample['trg_kps'].numpy()
            trg_bbox = sample['trg_bbox']

            pred_matches = []

            # iterate over keypoints
            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]
                patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size=patch_size, resized_size=img_size)

                # extract source feature
                src_feature = src_features[0, patch_y, patch_x, :]

                # compute cosine similarities
                similarities = F.cosine_similarity(
                    src_feature.unsqueeze(0),
                    tgt_flat,
                    dim=1
                )

                # find best match with windowed softargmax
                match_patch_x, match_patch_y = find_best_match_window_softargmax(
                    similarities, W, H, K=K, temperature=temperature
                )
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size,
                    patch_size=patch_size, resized_size=img_size
                )

                pred_matches.append([match_x, match_y])

            # compute PCK for each threshold
            image_pcks = {}
            for threshold in thresholds:
                pck, _, _ = compute_pck_spair71k(
                    pred_matches,
                    trg_kps.tolist(),
                    trg_bbox,
                    threshold
                )
                image_pcks[threshold] = pck

            per_image_metrics.append({
                'category': sample['category'],
                'num_keypoints': src_kps.shape[0],
                'pck_scores': image_pcks,
            })
            if idx==100 or idx%1000==0:
                print(f"  Processed {idx+1}/{len(dataset)} images", flush=True)
            # if idx==10:
            #     break  # debug test on 50 images only

    return per_image_metrics

def run_grid_search(model, val_dataset, device, results_dir):
    """Run grid search over K and temperature parameters."""

    #hyperparameter ranges
    K_values = [3, 5, 7, 9, 11]
    #K = 5
    temperature_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    thresholds = [0.05, 0.1, 0.2]

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GRID SEARCH FOR WINDOWED SOFTARGMAX HYPERPARAMETERS")
    print("=" * 80)
    # print(f"K values: {K_values}")
    print(f"Temperature values: {temperature_values}")
    print(f"Total combinations: {len(K_values) * len(temperature_values)}")
    print(f"Validation set size: {len(val_dataset)}")
    print("=" * 80)

    all_results = []
    total_combinations = len(K_values) * len(temperature_values)
    current_combo = 0

    for K in K_values:
        for temp in temperature_values:
            current_combo += 1
            print(f"\n[{current_combo}/{total_combinations}] Testing K={K}, temperature={temp}")

            start_time = time.time()
            per_image_metrics = evaluate_with_params(
                model, val_dataset, device, K, temp, thresholds
            )
            inference_time = time.time() - start_time

            result = {
                'K': K,
                'temperature': temp,
                'inference_time_sec': inference_time,
            }

            for threshold in thresholds:
                all_pcks = np.array([img['pck_scores'][threshold] for img in per_image_metrics])

                result[f'pck@{threshold:.2f}_mean'] = float(np.mean(all_pcks))
                result[f'pck@{threshold:.2f}_std'] = float(np.std(all_pcks))
                result[f'pck@{threshold:.2f}_median'] = float(np.median(all_pcks))
                result[f'pck@{threshold:.2f}_p25'] = float(np.percentile(all_pcks, 25))
                result[f'pck@{threshold:.2f}_p75'] = float(np.percentile(all_pcks, 75))

                print(f"  PCK@{threshold:.2f}: mean={result[f'pck@{threshold:.2f}_mean']:.2f}%, "
                      f"median={result[f'pck@{threshold:.2f}_median']:.2f}%")

            all_results.append(result)
            print(f"  Time: {inference_time:.2f}s")

    #save all results_SPair71K to CSV
    df_results = pd.DataFrame(all_results)
    csv_path = f'{results_dir}/grid_search_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Saved grid search results to '{csv_path}'")

    #find best parameters for each threshold
    print(f"\n{'=' * 80}")
    print("BEST PARAMETERS FOR EACH THRESHOLD")
    print("=" * 80)

    best_params_summary = []
    for threshold in thresholds:
        metric_col = f'pck@{threshold:.2f}_mean'
        best_idx = df_results[metric_col].idxmax()
        best_row = df_results.loc[best_idx]

        best_params = {
            'threshold': threshold,
            'best_K': int(best_row['K']),
            'best_temperature': float(best_row['temperature']),
            'best_pck_mean': float(best_row[metric_col]),
            'best_pck_median': float(best_row[f'pck@{threshold:.2f}_median']),
            'best_pck_std': float(best_row[f'pck@{threshold:.2f}_std']),
        }
        best_params_summary.append(best_params)

        print(f"\nPCK@{threshold:.2f}:")
        print(f"  Best K: {best_params['best_K']}")
        print(f"  Best temperature: {best_params['best_temperature']}")
        print(f"  Mean PCK: {best_params['best_pck_mean']:.2f}%")
        print(f"  Median PCK: {best_params['best_pck_median']:.2f}%")
        print(f"  Std PCK: {best_params['best_pck_std']:.2f}%")


    df_best = pd.DataFrame(best_params_summary)
    best_csv_path = f'{results_dir}/best_parameters.csv'
    df_best.to_csv(best_csv_path, index=False)
    print(f"\nSaved best parameters to '{best_csv_path}'")


    best_json_path = f'{results_dir}/best_parameters.json'
    with open(best_json_path, 'w') as f:
        json.dump(best_params_summary, f, indent=2)
    print(f"Saved best parameters to '{best_json_path}'")

    print("=" * 80)

    return df_results, best_params_summary


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 512
    patch_size = 16
    # model = vit_base(
    #     img_size=(img_size, img_size),  # base / nominal size
    #     patch_size=patch_size,  # patch size that matches the checkpoint
    #     num_register_tokens=0,  # <- no registers
    #     block_chunks=0,
    #     init_values=1.0,  # LayerScale initialization
    # )
    model = vit_base (
        img_size= (img_size, img_size),        # base / nominal size
        patch_size=patch_size,             # patch size that matches the checkpoint
        n_storage_tokens=4,
        layerscale_init= 1.0,
        mask_k_bias=True,
    )    
    ckpt = torch.load("models/dinov3/weights/finetuned/dinov3_vitb16_finetuned_3bl_0.0001lr_15t.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    base = 'Spair71k'
    pair_ann_path = f'{base}/PairAnnotation'
    layout_path = f'{base}/Layout'
    image_path = f'{base}/JPEGImages'
    dataset_size = 'large'
    pck_alpha = 0.1  # mock, it's not used in evaluation
    val_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='val')

    df_results, best_params = run_grid_search(model, val_dataset, device, 'grid_search_results/dinov3/dinov3_finetuned')