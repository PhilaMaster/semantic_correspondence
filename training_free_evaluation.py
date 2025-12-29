import torch
from SPair71k.devkit.SPairDataset import SPairDataset
from evaluate import evaluate, save_results
from models.dinov2.dinov2.models.vision_transformer import vit_base, vit_small, vit_large
import os
from datetime import datetime

#parameters
thresholds = [0.05, 0.1, 0.2]

base = 'Spair71k'
pair_ann_path = f'{base}/PairAnnotation'
layout_path = f'{base}/Layout'
image_path = f'{base}/JPEGImages'
dataset_size = 'large'
pck_alpha = 0.1 #mock, it's not used in evaluation
use_windowed_softargmax = False


#results folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results/dinov2_base_spair71k_finetuned_1epoch_2blocks'
results_dir+= '_wsoftargmax_' if use_windowed_softargmax else '_argmax_'
results_dir+=timestamp
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")


#patch size that matches the checkpoint (14 for vitb14)
model = vit_base(
    img_size=(518, 518),        # base / nominal size
    patch_size=14,             # patch size that matches the checkpoint
    num_register_tokens=0,     # <- no registers
    block_chunks=0,
    init_values=1.0,  # LayerScale initialization
)

device = "cuda" if torch.cuda.is_available() else "cpu" #use GPU if available
print("Using device:", device)
ckpt = torch.load("models/dinov2/dinov2_training_checkpoint_epoch1_10temp.pth", map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.to(device)
model.eval()


test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='test')

per_image_metrics, all_keypoint_metrics, total_inference_time_sec = evaluate(
    model,
    test_dataset,
    device,
    thresholds,
    use_windowed_softargmax,
    K=5,
    temperature=0.1
)

save_results(per_image_metrics, all_keypoint_metrics, results_dir, total_inference_time_sec, thresholds)