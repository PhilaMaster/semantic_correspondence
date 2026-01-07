import json
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPair71k.devkit.SPairDataset import SPairDataset
from helper_functions import extract_dense_features, pixel_to_patch_coord, patch_to_pixel_coord
from finetuning.simple_eval import simple_evaluate
from matching_strategies import find_best_match_argmax
from pck import compute_pck_spair71k
# from models.dinov2.dinov2.models.vision_transformer import vit_base
from models.dinov3.dinov3.models.vision_transformer import vit_base
from finetune_dinov2 import freeze_model, unfreeze_last_n_blocks, train_epoch


def main():
    """Main training and evaluation pipeline"""

    # ========== CONFIGURATION ==========
    n_blocks = 2  #to try: 1, 2, 3, 4
    num_epochs = 1
    learning_rate = 1e-4
    batch_size = 1  #SPair-71k has variable-sized images
    temperature = 1  #softmax temperature for cross-entropy loss
    img_size = 512
    patch_size = 16
    weight_decay = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results_SPair71K directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'temperature_comparison/dinov3/t_{temperature}_blocks_{n_blocks}_{timestamp}'
    # results_dir = f'results_SPair71K/dinov3_base_finetuned_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # ========== LOAD DATASETS ==========
    print("\nLoading SPair-71k dataset...")
    base = '../Spair71k'
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
    print("\nLoading DINOv3-base model...")
    model = vit_base(
        img_size= (img_size, img_size),        # base / nominal size
        patch_size=patch_size,             # patch size that matches the checkpoint
        n_storage_tokens=4,
        layerscale_init= 1.0,
        mask_k_bias=True,
    )

    # load pretrained weights
    ckpt = torch.load("../models/dinov3/weights/dinov3_vitb16_pretrain.pth", map_location=device)
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
        weight_decay=weight_decay
    )

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
            model, train_loader, optimizer, device, epoch + 1, img_size, patch_size, temperature=temperature
        )
        print(f"\nAverage training loss: {train_loss:.4f}")

        # Update learning rate
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Validate on val set
        print("\nEvaluating on test set...")
        results_val, per_image_metrics = simple_evaluate(model, val_dataset, device, img_size, patch_size)

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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'n_blocks': n_blocks,
            'temperature': temperature,
            'learning_rate': learning_rate,
            'val_pck@0.05': pck_005,
            'val_pck@0.10': pck_010,
            'val_pck@0.20': pck_020,
        }, ckpt_path)
        print(f"✓ Checkpoint saved: {ckpt_path}")

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
        #     print(f"✓ Best model saved: {best_ckpt_path}")

        # Store training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            'val_pck@0.05': pck_005,
            'val_pck@0.10': pck_010,
            'val_pck@0.20': pck_020,
        })

        # Save intermediate results_SPair71K
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
        print(f"✓ Metadata saved: {results_dir}/metadata.json")

    


if __name__ == "__main__":
    main()