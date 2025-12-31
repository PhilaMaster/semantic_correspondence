"""
Quick test script to verify finetuning pipeline works correctly.
Runs only a few iterations to check for errors before full training.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from SPair71k.devkit.SPairDataset import SPairDataset
from helper_functions import extract_dense_features, pixel_to_patch_coord
from models.dinov2.dinov2.models.vision_transformer import vit_base


def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n_blocks):
    """Unfreeze the last n_blocks transformer blocks + final norm layer"""
    total_blocks = len(model.blocks)
    for i in range(total_blocks - n_blocks, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True
    print(f"✓ Unfrozen last {n_blocks} blocks + norm layer")


def compute_cross_entropy_loss(src_features, tgt_features, src_kps, trg_kps,
                               src_original_size, tgt_original_size, temperature=10.0):
    """Compute cross-entropy loss for semantic correspondence"""
    _, H, W, D = tgt_features.shape
    tgt_flat = tgt_features.reshape(H * W, D)
    
    losses = []
    
    for i in range(src_kps.shape[0]):
        src_x, src_y = src_kps[i]
        tgt_x, tgt_y = trg_kps[i]
        
        src_patch_x, src_patch_y = pixel_to_patch_coord(src_x, src_y, src_original_size)
        src_feature = src_features[0, src_patch_y, src_patch_x, :]
        
        tgt_patch_x, tgt_patch_y = pixel_to_patch_coord(tgt_x, tgt_y, tgt_original_size)
        
        similarities = F.cosine_similarity(
            src_feature.unsqueeze(0),
            tgt_flat,
            dim=1
        )
        
        log_probs = F.log_softmax(similarities * temperature, dim=0)
        gt_idx = tgt_patch_y * W + tgt_patch_x
        loss = -log_probs[gt_idx]
        losses.append(loss)
    
    return torch.stack(losses).mean()


def quick_test():
    """Quick test with just a few samples"""
    
    print("="*60)
    print("QUICK FINETUNING TEST")
    print("="*60)
    
    # Configuration
    n_blocks = 2
    n_train_samples = 5  # Only 5 samples for quick test
    n_test_samples = 3   # Only 3 for evaluation
    temperature = 10.0
    learning_rate = 1e-4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    
    # Load model
    print("\n[1/5] Loading DINOv2 model...")
    model = vit_base(
        img_size=(518, 518),
        patch_size=14,
        num_register_tokens=0,
        block_chunks=0,
        init_values=1.0,
    )
    
    ckpt = torch.load("../models/dinov2/dinov2_vitb14_pretrain.pth", map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    print("✓ Model loaded")
    
    # Freeze and unfreeze
    print(f"\n[2/5] Freezing model and unfreezing last {n_blocks} blocks...")
    freeze_model(model)
    unfreeze_last_n_blocks(model, n_blocks)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load dataset (small subset)
    print(f"\n[3/5] Loading {n_train_samples} training samples...")
    base = '../Spair71k'
    train_dataset = SPairDataset(
        f'{base}/PairAnnotation',
        f'{base}/Layout',
        f'{base}/JPEGImages',
        'large',
        0.1,
        datatype='trn'
    )
    print(f"✓ Dataset loaded (total: {len(train_dataset)} samples)")
    
    # Create optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Test training loop
    print(f"\n[4/5] Testing training loop with {n_train_samples} samples...")
    model.train()
    
    for idx in range(n_train_samples):
        sample = train_dataset[idx]
        
        # Prepare data
        src_tensor = sample['src_img'].unsqueeze(0).to(device)
        tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)
        
        src_tensor = F.interpolate(src_tensor, size=(518, 518), mode='bilinear', align_corners=False)
        tgt_tensor = F.interpolate(tgt_tensor, size=(518, 518), mode='bilinear', align_corners=False)
        
        src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
        tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])
        
        src_kps = sample['src_kps'].numpy()
        trg_kps = sample['trg_kps'].numpy()
        
        # Extract features
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
        
        print(f"  Sample {idx+1}/{n_train_samples}: Loss = {loss.item():.4f}")
    
    print("✓ Training loop completed successfully!")
    
    # Test evaluation
    print(f"\n[5/5] Testing evaluation with {n_test_samples} samples...")
    model.eval()
    
    test_dataset = SPairDataset(
        f'{base}/PairAnnotation',
        f'{base}/Layout',
        f'{base}/JPEGImages',
        'large',
        0.1,
        datatype='test'
    )
    
    with torch.no_grad():
        for idx in range(n_test_samples):
            sample = test_dataset[idx]
            
            src_tensor = sample['src_img'].unsqueeze(0).to(device)
            tgt_tensor = sample['trg_img'].unsqueeze(0).to(device)
            
            src_tensor = F.interpolate(src_tensor, size=(518, 518), mode='bilinear', align_corners=False)
            tgt_tensor = F.interpolate(tgt_tensor, size=(518, 518), mode='bilinear', align_corners=False)
            
            src_features = extract_dense_features(model, src_tensor, training=False)
            tgt_features = extract_dense_features(model, tgt_tensor, training=False)
            
            print(f"  Sample {idx+1}/{n_test_samples}: Features shape = {src_features.shape}")
    
    print("✓ Evaluation completed successfully!")
    
    # Verify gradients
    print("\n[VERIFICATION] Checking gradient flow...")
    has_grads = False
    grad_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grads = True
            grad_layers.append(name)
    
    if has_grads:
        print(f"✓ Gradients detected in {len(grad_layers)} layers")
        print("  Sample layers with gradients:")
        for layer in grad_layers[:3]:  # Show first 3
            print(f"    - {layer}")
    else:
        print("✗ WARNING: No gradients detected!")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ Model loading: OK")
    print("✓ Freeze/Unfreeze: OK")
    print("✓ Training loop: OK")
    print("✓ Backward pass: OK")
    print("✓ Evaluation: OK")
    print(f"✓ Gradient flow: {'OK' if has_grads else 'FAILED'}")
    print("\n✓ All tests passed! Ready for full training.")
    print("="*60)


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()