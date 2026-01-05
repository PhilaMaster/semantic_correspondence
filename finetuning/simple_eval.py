import numpy as np
import torch
import torch.nn.functional as F

from helper_functions import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from matching_strategies import find_best_match_argmax
from pck import compute_pck_spair71k


def simple_evaluate(model, dataset, device, thresholds=[0.05, 0.1, 0.2]):
    """
    Evaluate model on test set using PCK metric

    Args:
        model: DINOv2 model
        dataset: test dataset
        device: 'cuda' or 'cpu'
        thresholds: list of PCK thresholds to evaluate

    Returns:
        results_SPair71K: dictionary with PCK scores at different thresholds
    """
    model.eval()
    per_image_metrics = []

    print(f"Evaluating on {len(dataset)} image pairs...")

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            # Prepare images
            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)

            src_tensor = F.interpolate(src_tensor, size=(518, 518), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(518, 518), mode='bilinear', align_corners=False)

            src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
            tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

            # Extract features
            src_features = extract_dense_features(model, src_tensor)
            tgt_features = extract_dense_features(model, tgt_tensor)

            _, H, W, D = tgt_features.shape
            tgt_flat = tgt_features.reshape(H * W, D)

            # Get keypoints and bbox
            src_kps = sample['src_kps'].numpy()
            trg_kps = sample['trg_kps'].numpy()
            trg_bbox = sample['trg_bbox']

            # Predict matches for all keypoints
            pred_matches = []

            for i in range(src_kps.shape[0]):
                src_x, src_y = src_kps[i]

                # Get source feature
                patch_x, patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size)
                src_feature = src_features[0, patch_y, patch_x, :]

                # Compute similarities
                similarities = F.cosine_similarity(
                    src_feature.unsqueeze(0),
                    tgt_flat,
                    dim=1
                )

                # Find best match using argmax
                match_patch_x, match_patch_y = find_best_match_argmax(similarities, W)
                match_x, match_y = patch_to_pixel_coord(
                    match_patch_x, match_patch_y, tgt_original_size
                )

                pred_matches.append([match_x, match_y])

            # Compute PCK for different thresholds
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
                'pck_scores': image_pcks,
            })

            if (idx + 1) % 100 == 0:
                print(f"Evaluated {idx + 1}/{len(dataset)} pairs...")

    # Compute overall statistics
    results = {}
    for threshold in thresholds:
        all_pcks = [img['pck_scores'][threshold] for img in per_image_metrics]
        results[f'pck@{threshold:.2f}'] = {
            'mean': float(np.mean(all_pcks)),
            'std': float(np.std(all_pcks)),
            'median': float(np.median(all_pcks)),
        }

    return results, per_image_metrics