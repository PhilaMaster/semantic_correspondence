import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np

# Assicurati che i path siano corretti rispetto alla tua struttura cartelle
from SPair71k.devkit.SPairDataset import SPairDataset
from helper_functions import pixel_to_patch_coord
# Importiamo la loss function dal tuo script esistente per coerenza
from finetune_dinov2 import compute_cross_entropy_loss
from finetuning.simple_eval import simple_evaluate

# Import SAM
from models.segment_anything.segment_anything import sam_model_registry

def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_sam_image_encoder(sam_model, n_blocks):
    """
    Unfreeze the last n_blocks transformer blocks of the SAM Image Encoder + Neck.
    """
    # L'image encoder di SAM è solitamente un ViT
    image_encoder = sam_model.image_encoder
    total_blocks = len(image_encoder.blocks)

    print(f"SAM Image Encoder has {total_blocks} blocks.")

    # 1. Unfreeze last n blocks
    for i in range(total_blocks - n_blocks, total_blocks):
        for param in image_encoder.blocks[i].parameters():
            param.requires_grad = True
    
    # 2. Unfreeze the NECK (Importante per SAM: proietta le feature a 256 dim)
    # Il neck è composto da conv layers che elaborano l'output del ViT
    for param in image_encoder.neck.parameters():
        param.requires_grad = True

    print(f"Unfrozen last {n_blocks} blocks + Neck of SAM Image Encoder")

def extract_sam_features(sam_model, img_tensor):
    """
    Extract dense features from SAM Image Encoder.
    SAM output format is usually [B, 256, H/16, W/16].
    We permute it to [B, H/16, W/16, 256] for the loss function.
    """
    # SAM si aspetta l'input normalizzato. 
    # Assumiamo che il dataloader fornisca già immagini normalizzate (o lo faccia SAM internamente se si usa il preprocess).
    # Qui usiamo direttamente l'image_encoder che si aspetta tensori [B, 3, H, W].
    
    features = sam_model.image_encoder(img_tensor) # Output: [B, 256, 32, 32] se input 512
    
    # Permute per avere [B, H, W, D] come si aspetta la funzione compute_cross_entropy_loss
    return features.permute(0, 2, 3, 1)

def train_epoch_sam(sam_model, dataloader, optimizer, device, epoch, img_size, patch_size=16, temperature=10.0):
    sam_model.image_encoder.train() # Mettiamo in train solo l'encoder (se necessario)
    # Nota: se hai freezato tutto il resto, .train() globale va bene, ma sii specifico se usi BatchNorm
    
    total_loss = 0
    num_batches = 0

    for idx, sample in enumerate(dataloader):
        # Prepare data
        src_tensor = sample['src_img'].to(device)  # [1, 3, H, W]
        tgt_tensor = sample['trg_img'].to(device)  # [1, 3, H, W]

        # Resize (SAM usually works ideally at 1024, but 512 works for fine-tuning)
        src_tensor = F.interpolate(src_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        tgt_tensor = F.interpolate(tgt_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # Store original sizes for coordinate conversion
        src_original_size = (sample['src_imsize'][2], sample['src_imsize'][1])
        tgt_original_size = (sample['trg_imsize'][2], sample['trg_imsize'][1])

        # Get keypoints
        src_kps = sample['src_kps'].numpy()[0]
        trg_kps = sample['trg_kps'].numpy()[0]

        # Extract dense features using SAM
        src_features = extract_sam_features(sam_model, src_tensor)
        tgt_features = extract_sam_features(sam_model, tgt_tensor)

        # Compute loss
        # patch_size per SAM è relativo allo stride finale (che è 16 in SAM standard)
        loss = compute_cross_entropy_loss(
            src_features, tgt_features,
            src_kps, trg_kps,
            src_original_size, tgt_original_size,
            img_size, patch_size, 
            temperature=temperature
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (idx + 1) % 50 == 0:
            print(f"Epoch {epoch}, Batch {idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches

def main():
    # ========== CONFIGURATION ==========
    model_type = "vit_b"
    checkpoint_path = "../models/segment_anything/weights/sam_vit_b_01ec64.pth" 
    
    n_blocks_to_unfreeze = 2
    num_epochs = 1
    learning_rate = 1e-4 # SAM potrebbe gradire un LR più basso, prova 1e-5 se instabile
    temperature = 10
    img_size = 512
    patch_size = 16 # Stride di SAM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_SPair71K/SAM_finetuned_{model_type}_{n_blocks_to_unfreeze}blocks_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # ========== DATASET ==========
    print("\nLoading SPair-71k dataset...")
    base = '../Spair71k'
    train_dataset = SPairDataset(
        f'{base}/PairAnnotation', f'{base}/Layout', f'{base}/JPEGImages',
        'large', 0.1, datatype='trn'
    )
    # Per semplicità uso train anche come val per test veloce codice, cambia con 'val'
    val_dataset = SPairDataset(
        f'{base}/PairAnnotation', f'{base}/Layout', f'{base}/JPEGImages',
        'large', 0.1, datatype='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # ========== MODEL ==========
    print(f"\nLoading SAM ({model_type})...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    # Freeze tutto e sblocca ultimi blocchi encoder
    freeze_model(sam)
    unfreeze_sam_image_encoder(sam, n_blocks_to_unfreeze)

    # Parametri allenabili
    trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer (passiamo solo i parametri che richiedono grad)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    # ========== LOOP ==========
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        avg_loss = train_epoch_sam(
            sam, train_loader, optimizer, device, epoch+1, 
            img_size=img_size, patch_size=patch_size, temperature=temperature
        )
        print(f"Average Loss: {avg_loss:.4f}")

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': sam.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'n_blocks': n_blocks_to_unfreeze,
        }, f'{results_dir}/sam_finetuned_epoch{epoch+1}.pth')
        
        print("Checkpoint saved.")

    print(f"Training completed. Results in {results_dir}")

if __name__ == "__main__":
    main()