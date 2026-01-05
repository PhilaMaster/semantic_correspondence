import json
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import torch.nn.functional as F

from SPair71k.devkit.SPairDataset import SPairDataset
from helper_functions import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from finetuning.simple_eval import simple_evaluate
from matching_strategies import find_best_match_argmax
from pck import compute_pck_spair71k
from models.dinov2.dinov2.models.vision_transformer import vit_base


def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n_blocks):
    """
    Unfreeze the last n_blocks transformer blocks + final norm layer

    Args:
        model: DINOv2 model
        n_blocks: number of blocks to unfreeze (counting from the end)
    """
    total_blocks = len(model.blocks)

    # Unfreeze last n blocks
    for i in range(total_blocks - n_blocks, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Also unfreeze the final normalization layer
    for param in model.norm.parameters():
        param.requires_grad = True

    print(f"Unfrozen last {n_blocks} blocks + norm layer")


def compute_cross_entropy_loss(src_features, tgt_features, src_kps, trg_kps,
                               src_original_size, tgt_original_size, temperature=10.0):
    """
    Compute cross-entropy loss for semantic correspondence.
    Treats correspondence as a classification problem where each target patch is a class.

    Args:
        src_features: [1, H, W, D] source dense features
        tgt_features: [1, H, W, D] target dense features
        src_kps: [N, 2] source keypoints in pixel coordinates
        trg_kps: [N, 2] target keypoints in pixel coordinates
        src_original_size: (width, height) of original source image
        tgt_original_size: (width, height) of original target image
        temperature: softmax temperature (higher = more peaked distribution)

    Returns:
        loss: mean cross-entropy loss across all keypoints
    """
    _, H, W, D = tgt_features.shape
    tgt_flat = tgt_features.reshape(H * W, D)  # [H*W, D]

    losses = []

    for i in range(src_kps.shape[0]):
        src_x, src_y = src_kps[i]
        tgt_x, tgt_y = trg_kps[i]

        # Get source feature at keypoint location
        src_patch_x, src_patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size)
        src_feature = src_features[0, src_patch_y, src_patch_x, :]  # [D]

        # Get ground truth target patch coordinates
        tgt_patch_x, tgt_patch_y = pixel_to_patch_coord(tgt_x, tgt_y, tgt_original_size)

        # Compute cosine similarities with all target patches
        similarities = F.cosine_similarity(
            src_feature.unsqueeze(0),  # [1, D]
            tgt_flat,  # [H*W, D]
            dim=1
        )  # [H*W]

        # Convert similarities to log-probabilities
        log_probs = F.log_softmax(similarities * temperature, dim=0)

        # Ground truth index (flatten 2D coordinates to 1D)
        gt_idx = tgt_patch_y * W + tgt_patch_x

        # Negative log-likelihood loss
        loss = -log_probs[gt_idx]
        losses.append(loss)

    return torch.stack(losses).mean()


def train_epoch(model, dataloader, optimizer, device, epoch, temperature=10.0):
    """
    Train for one epoch

    Args:
        model: DINOv2 model
        dataloader: training data loader
        optimizer: optimizer
        device: 'cuda' or 'cpu'
        epoch: current epoch number
        temperature: softmax temperature for loss

    Returns:
        avg_loss: average loss over the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for idx, sample in enumerate(dataloader):
        # Prepare data
        src_tensor = sample['src_img'].to(device)  # [1, 3, H, W]
        tgt_tensor = sample['trg_img'].to(device)  # [1, 3, H, W]

        # Resize to 518x518 (DINOv2 expects this size)
        src_tensor = F.interpolate(src_tensor, size=(518, 518), mode='bilinear', align_corners=False)
        tgt_tensor = F.interpolate(tgt_tensor, size=(518, 518), mode='bilinear', align_corners=False)

        # Store original sizes for coordinate conversion
        src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
        tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

        # Get keypoints
        src_kps = sample['src_kps'].numpy()[0]  # [N, 2]
        trg_kps = sample['trg_kps'].numpy()[0]  # [N, 2]

        # Extract dense features
        src_features = extract_dense_features(model, src_tensor, training=True)
        tgt_features = extract_dense_features(model, tgt_tensor, training=True)

        # Compute loss
        loss = compute_cross_entropy_loss(
            src_features, tgt_features,
            src_kps, trg_kps,
            src_original_size, tgt_original_size,
            temperature=temperature
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if (idx + 1) % 50 == 0:
            print(f"Epoch {epoch}, Batch {idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training and evaluation pipeline"""

    # ========== CONFIGURATION ==========
    # n_blocks_to_unfreeze = 1  #to try: 1, 2, 3, 4
    num_epochs = 1
    learning_rate = 1e-4
    batch_size = 1  #SPair-71k has variable-sized images
    temperature = 10  #softmax temperature for cross-entropy loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results_SPair71K directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_dir = f'results_SPair71K/dinov2_base_finetuned_{n_blocks_to_unfreeze}blocks_{timestamp}'
    results_dir = f'results_dinov2/ts_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # ========== LOAD DATASETS ==========
    print("\nLoading SPair-71k dataset...")
    base = '../Spair71k'

    train_dataset = SPairDataset(
        f'{base}/PairAnnotation',
        f'{base}/Layout',
        f'{base}/JPEGImages',
        'large',
        0.1,  # dummy pck_alpha, not used during training
        datatype='trn'  # training split
    )

    val_dataset = SPairDataset(
        f'{base}/PairAnnotation',
        f'{base}/Layout',
        f'{base}/JPEGImages',
        'large',
        0.1,
        datatype='val'
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if device == 'cuda' else False
    )

    for n_blocks in [1,2,3,4]:
        print("\n" + "=" * 80)
        print(f"FINETUNING WITH LAST {n_blocks} BLOCKS UNFROZEN")
        print("=" * 80)
        # ========== LOAD MODEL ==========
        print("\nLoading DINOv2-base model...")
        model = vit_base(
            img_size=(518, 518),
            patch_size=14,
            num_register_tokens=0,
            block_chunks=0,
            init_values=1.0,
        )

        # load pretrained weights
        ckpt = torch.load("../models/dinov2/dinov2_vitb14_pretrain.pth", map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.to(device)

        # freeze entire model, then unfreeze last N blocks
        freeze_model(model)
        unfreeze_last_n_blocks(model, n_blocks)

        # count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")


        # ========== OPTIMIZER ==========
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Optional: Learning rate scheduler
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # ========== TRAINING LOOP ==========
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        # best_pck = 0
        # best_epoch = 0
        # training_history = []

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print('=' * 60)

            # Train for one epoch
            train_loss = train_epoch(
                model, train_loader, optimizer, device, epoch + 1, temperature=temperature
            )
            print(f"\nAverage training loss: {train_loss:.4f}")

            # Update learning rate
            # scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

            # Evaluate on test set
            # print("\nEvaluating on test set...")
            # results_SPair71K, per_image_metrics = simple_evaluate(model, val_dataset, device)

            # print("\nTest Results:")
            # for key, value in results_SPair71K.items():
            #     print(f"  {key}: {value['mean']:.2f}% ± {value['std']:.2f}% "
            #           f"(median: {value['median']:.2f}%)")

            # Save best model
            # current_pck = results_SPair71K['pck@0.10']['mean']
            # if current_pck > best_pck:
            #     best_pck = current_pck
            #     best_epoch = epoch + 1

            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'pck': best_pck,
                'n_blocks': n_blocks,
            }, f'{results_dir}/10temp_{n_blocks}_blocks_1epoch.pth.pth')

            # print(f"\n✓ New best model saved! PCK@0.1: {best_pck:.2f}%")
            print(f"\n✓ Finetuning finished for {n_blocks} unfreezed model")
            # Store training history
            # training_history.append({
            #     'epoch': epoch + 1,
            #     'train_loss': train_loss,
            #     'learning_rate': current_lr,
                # 'test_results': results_SPair71K
            # })

            # Save intermediate results_SPair71K
            # with open(f'{results_dir}/training_history.json', 'w') as f:
            #     json.dump(training_history, f, indent=2)

        # ========== FINAL RESULTS ==========
        # print("\n" + "=" * 60)
        # print("TRAINING COMPLETED")
        # print("=" * 60)
        # print(f"Best PCK@0.1: {best_pck:.2f}% (Epoch {best_epoch})")
        # print(f"Results saved to: {results_dir}")
        #
        # # Load best model and evaluate on full test set
        # print("\nLoading best model for final evaluation...")
        # checkpoint = torch.load(f'{results_dir}/best_model.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        #
        # final_results, final_per_image = evaluate(model, test_dataset, device)

        # Save final detailed results_SPair71K
        # with open(f'{results_dir}/final_results.json', 'w') as f:
        #     json.dump({
        #         'best_epoch': best_epoch,
        #         'n_blocks_unfrozen': n_blocks_to_unfreeze,
        #         'temperature': temperature,
        #         'learning_rate': learning_rate,
        #         'num_epochs': num_epochs,
        #         # 'results_SPair71K': final_results
        #     }, f, indent=2)

    # Save per-category analysis
    # category_results = defaultdict(lambda: defaultdict(list))
    # for img_metric in final_per_image:
    #     category = img_metric['category']
    #     for threshold, pck in img_metric['pck_scores'].items():
    #         category_results[category][threshold].append(pck)

    # Compute per-category statistics
    # category_stats = {}
    # for category, thresholds_dict in category_results.items():
    #     category_stats[category] = {}
    #     for threshold, pcks in thresholds_dict.items():
    #         category_stats[category][f'pck@{threshold:.2f}'] = {
    #             'mean': float(np.mean(pcks)),
    #             'std': float(np.std(pcks)),
    #             'n_samples': len(pcks)
    #         }

    # with open(f'{results_dir}/per_category_results.json', 'w') as f:
    #     json.dump(category_stats, f, indent=2)

    # print("\nPer-category results_SPair71K:")
    # for category, stats in sorted(category_stats.items()):
    #     pck_01 = stats['pck@0.10']['mean']
    #     n_samples = stats['pck@0.10']['n_samples']
    #     print(f"  {category:20s}: {pck_01:.2f}% (n={n_samples})")
    #
    # print("\n" + "=" * 60)


if __name__ == "__main__":
    main()