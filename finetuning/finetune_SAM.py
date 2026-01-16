import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPair71k.devkit.SPairDataset import SPairDataset
from helper_functions import pixel_to_patch_coord, extract_dense_features_SAM
from finetune_dinov2 import compute_cross_entropy_loss
from finetuning.simple_eval import simple_evaluate_SAM
from models.segment_anything.segment_anything import sam_model_registry

def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n_blocks):
    """
    Unfreeze the last n_blocks transformer blocks + final norm layer of the SAM image encoder.

    Args:
        model: SAM model
        n_blocks: number of blocks to unfreeze (counting from the end)
    """
    # Access the image encoder part of the SAM model
    image_encoder = model.image_encoder

    total_blocks = len(image_encoder.blocks)

    # Unfreeze last n blocks
    for i in range(total_blocks - n_blocks, total_blocks):
        for param in image_encoder.blocks[i].parameters():
            param.requires_grad = True

    # Also unfreeze the final normalization layer
    # For SAM's ViT, this is typically model.image_encoder.neck.ln_final
    if hasattr(image_encoder, 'neck') and hasattr(image_encoder.neck, 'ln_final'):
        for param in image_encoder.neck.ln_final.parameters():
            param.requires_grad = True
        print(f"Unfrozen last {n_blocks} blocks + final norm layer of SAM image encoder")
    else:
        print(f"Unfrozen last {n_blocks} blocks of SAM image encoder. Final norm layer not found or accessible via 'neck.ln_final'.")


def compute_cross_entropy_loss(src_features, tgt_features, src_kps, trg_kps,
                               src_original_size, tgt_original_size, img_size, patch_size, temperature=10.0):
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
        img_size: resizing size used during feature extraction
        patch_size: size of each patch
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
        src_patch_x, src_patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size, patch_size=patch_size, resized_size=img_size)
        src_feature = src_features[0, src_patch_y, src_patch_x, :]  # [D]

        # Get ground truth target patch coordinates
        tgt_patch_x, tgt_patch_y = pixel_to_patch_coord(tgt_x, tgt_y, tgt_original_size, patch_size=patch_size, resized_size=img_size)
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


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, img_size, patch_size, temperature=10.0):
    """
    Train for one epoch with Automatic Mixed Precision (AMP).

    Args:
        model: SAM model
        dataloader: training data loader
        optimizer: optimizer
        scaler: torch.cuda.amp.GradScaler for mixed precision
        device: 'cuda' or 'cpu'
        epoch: current epoch number
        img_size: size to which images are resized for feature extraction
        patch_size: size of each patch
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
        src_tensor = F.interpolate(src_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        tgt_tensor = F.interpolate(tgt_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # Store original sizes for coordinate conversion
        src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
        tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

        # Get keypoints
        src_kps = sample['src_kps'].numpy()[0]  # [N, 2]
        trg_kps = sample['trg_kps'].numpy()[0]  # [N, 2]

        optimizer.zero_grad()

        # Autocast operations to appropriate precision
        with torch.cuda.amp.autocast():
            # Extract dense features
            src_features = extract_dense_features_SAM(model, src_tensor, image_size=img_size, training=True)
            tgt_features = extract_dense_features_SAM(model, tgt_tensor, image_size=img_size, training=True)

            # Compute loss
            loss = compute_cross_entropy_loss(
                src_features, tgt_features,
                src_kps, trg_kps,
                src_original_size, tgt_original_size,
                img_size, patch_size,
                temperature=temperature
            )

        # Backward pass with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if (idx + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training and evaluation pipeline"""

    # ========== CONFIGURATION ==========+
    n_blocks = 3  #to try: 1, 2, 3, 4
    num_epochs = 1
    learning_rate = 1e-4
    batch_size = 1  #SPair-71k has variable-sized images
    temperature = 15  #softmax temperature for cross-entropy loss try 5,10,15
    img_size = 512
    patch_size = 16
    weight_decay = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results_SPair71k directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'blocks_comparison/SAM/lr_{learning_rate}_t_{temperature}_blocks_{n_blocks}_{timestamp}'
    # results_dir = f'results_SPair71k/dinov3_base_finetuned_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # ========== LOAD DATASETS ==========
    print("\nLoading SPair-71k dataset...")
    base = '../SPair71k'  #path to pf_pascal dataset
    pair_ann_path = f'{base}/PairAnnotation'
    layout_path = f'{base}/Layout'
    image_path = f'{base}/JPEGImages'
    dataset_size = 'large'
    pck_alpha = 0.1 #mock, it's not used in evaluation

    train_dataset = SPairDataset(
        pair_ann_path,
        layout_path,
        image_path,
        dataset_size,
        pck_alpha,  # dummy pck_alpha, not used during training
        datatype='trn'  # training split
    )

    val_dataset = SPairDataset(
        pair_ann_path,
        layout_path,
        image_path,
        dataset_size,
        pck_alpha,
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

    # for n_blocks in [1,2,3,4]:
    print("\n" + "=" * 80)
    print(f"FINETUNING WITH LAST {n_blocks} BLOCKS UNFROZEN")
    print("=" * 80)
    # ========== LOAD MODEL ==========
    print("\nLoading SAM model...")
    model_type = "vit_b"
    checkpoint_path = "../models/segment_anything/weights/sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam_model.to(device)

    # freeze entire model, then unfreeze last N blocks
    freeze_model(sam_model)
    unfreeze_last_n_blocks(sam_model, n_blocks)

    # count trainable parameters
    trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sam_model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)")


    # ========== OPTIMIZER AND GRAD SCALER ==========
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, sam_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Initialize GradScaler for Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ========== TRAINING LOOP ==========
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # best_pck = -1.0
    # best_epoch = -1
    training_history = []

    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('=' * 60)

        # Train for one epoch
        train_loss = train_epoch(
            sam_model, train_loader, optimizer, scaler, device, epoch + 1, img_size, patch_size, temperature=temperature
        )
        print(f"\nAverage training loss: {train_loss:.4f}")

        # Update learning rate
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Validate on val set
        print("\nEvaluating on test set...")
        # For evaluation, you typically don't use autocast as it's not about speed but accuracy.
        # However, for consistency, if the model was trained with AMP, it might be beneficial
        # to also run evaluation with autocast for feature extraction, but it's not strictly necessary.
        # We'll keep it as FP32 evaluation to ensure robustness.
        sam_model.eval() # Set model to evaluation mode
        with torch.no_grad(): # No gradients needed for evaluation
            results_val, per_image_metrics = simple_evaluate_SAM(sam_model, val_dataset, device, img_size, patch_size)

        pck_005 = results_val['pck@0.05']['mean']
        pck_010 = results_val['pck@0.10']['mean']
        pck_020 = results_val['pck@0.20']['mean']

        print(f"Val Results:")
        print(f"  PCK@0.05: {pck_005:.2f}%")
        print(f"  PCK@0.10: {pck_010:.2f}%")
        print(f"  PCK@0.20: {pck_020:.2f}%")


        # Save model checkpoint
        # Save checkpoint for this epoch
        ckpt_path = f'{results_dir}/epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': sam_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'n_blocks': n_blocks,
            'temperature': temperature,
            'learning_rate': learning_rate,
            'val_pck@0.05': pck_005,
            'val_pck@0.10': pck_010,
            'val_pck@0.20': pck_020,
        }, ckpt_path)
        print(f"\u2713 Checkpoint saved: {ckpt_path}")

        # Track best by PCK@0.10
        # if pck_010 > best_pck:
        #     best_pck = pck_010
        #     best_epoch = epoch + 1
        #     best_ckpt_path = f'{results_dir}/best_model.pth'
        #     torch.save({
        #         'epoch': best_epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'n_blocks': n_blocks,
        #         'temperature': temperature,
        #         'learning_rate': learning_rate,
        #         'val_pck@0.05': pck_005,
        #         'val_pck@0.10': pck_010,
        #         'val_pck@0.20': pck_020,
        #     }, best_ckpt_path)
        #     print(f"\u2713 Best model saved: {best_ckpt_path}")

        # Store training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            'val_pck@0.05': pck_005,
            'val_pck@0.10': pck_010,
            'val_pck@0.20': pck_020,
        })

        # Save intermediate results_SPair71k
        # with open(f'{results_dir}/training_history.json', 'w') as f:
        #     json.dump(training_history, f, indent=2)

        # ========== FINAL RESULTS ==========
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        # print(f"Best PCK@0.1: {best_pck:.2f}% (Epoch {best_epoch})")
        print(f"Results saved to: {results_dir}")


        # Save metadata for comparison
        metadata = {
            'n_blocks': n_blocks,
            'temperature': temperature,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            # 'best_epoch': best_epoch,
            # 'best_pck@0.05': float(training_history[best_epoch - 1]['val_pck@0.05']),
            # 'best_pck@0.10': float(best_pck),
            # 'best_pck@0.20': float(training_history[best_epoch - 1]['val_pck@0.20']),
            'pck@0.05': float(training_history[-1]['val_pck@0.05']),
            'pck@0.10': float(training_history[-1]['val_pck@0.10']),
            'pck@0.20': float(training_history[-1]['val_pck@0.20']),
            'training_history': training_history,
        }

        with open(f'{results_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\u2713 Metadata saved: {results_dir}/metadata.json")

if __name__ == "__main__":
    main()