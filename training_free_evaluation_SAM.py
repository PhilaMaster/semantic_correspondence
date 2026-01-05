import torch
from SPair71k.devkit.SPairDataset import SPairDataset
from evaluate import save_results
from evaluate_SAM import evaluate_SAM
from models.segment_anything.segment_anything import SamPredictor, sam_model_registry


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


model_type = "vit_b"  # o "vit_l" o "vit_b"
b = 'models/segment_anything/'
checkpoint_paths = {
    "vit_h": f"{b}sam_vit_h_4b8939.pth",
    "vit_l": f"{b}sam_vit_l_0b3195.pth",
    "vit_b": f"{b}sam_vit_b_01ec64.pth"
}

#results_SPair71K folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results_SPair71K/SAM/{model_type}'
results_dir+= '_wsoftargmax_' if use_windowed_softargmax else '_argmax_'
results_dir+=timestamp
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")


device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_paths[model_type])
sam.to(device)
sam.eval()
predictor = SamPredictor(sam)

device = "cuda" if torch.cuda.is_available() else "cpu" #use GPU if available
print("Using device:", device)


test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='test')

per_image_metrics, all_keypoint_metrics, total_inference_time_sec = evaluate_SAM(
    predictor,
    test_dataset,
    device,
    thresholds,
    use_windowed_softargmax,
    K=9,
    temperature=0.1
)

save_results(per_image_metrics, all_keypoint_metrics, results_dir, total_inference_time_sec, thresholds)