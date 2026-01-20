import torch
from SPair71k.devkit.SPairDataset import SPairDataset
# from pf_pascal.PFPascalDataset import PFPascalDataset
# from pf_willow.PFWillowDataset import PFWillowDataset
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
use_windowed_softargmax = True


model_type = "vit_b"  # o "vit_l" o "vit_b"
b = 'models/segment_anything/weights/'
checkpoint_paths = {
    "vit_h": f"{b}sam_vit_h_4b8939.pth",
    "vit_l": f"{b}sam_vit_l_0b3195.pth",
    "vit_b": f"{b}sam_vit_b_01ec64.pth"
}


#results_SPair71K folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results_PF_Willow/SAM/finetuned/{model_type}'
results_dir+= '_wsoftargmax_' if use_windowed_softargmax else '_argmax_'
results_dir+=timestamp
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")


device = "cuda" if torch.cuda.is_available() else "cpu" #use GPU if available
sam_model = sam_model_registry[model_type](checkpoint=checkpoint_paths[model_type])
sam_model.to(device)
sam_model.half()
sam_model.eval()
# predictor = SamPredictor(sam_model)
print("Using device:", device)
# checkpoint_path = "models/segment_anything/weights/finetuned/SAM_finetuned_4bl_15t_0.0001lr.pth"
# # Initialize the SAM model without loading checkpoint yet
# sam_model = sam_model_registry[model_type](checkpoint=None) # Pass None to initialize without loading
# sam_model.to(device)

# # Load the custom finetuned checkpoint
# print(f"Loading finetuned SAM checkpoint from {checkpoint_path}")
# checkpoint = torch.load(checkpoint_path, map_location=device)

# # The finetuned checkpoint likely contains more than just the model state_dict.
# # Extract the actual model_state_dict and load it
# if 'model_state_dict' in checkpoint:
#     sam_model.load_state_dict(checkpoint['model_state_dict'])
#     print("Successfully loaded 'model_state_dict' from checkpoint.")
# else:
#     # If the checkpoint itself is just the state_dict, try loading it directly
#     sam_model.load_state_dict(checkpoint)
#     print("Successfully loaded checkpoint directly as state_dict.")

test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='test')
# base = 'pf_willow'
# test_dataset = PFPascalDataset(base, split='test')
# test_dataset = PFWillowDataset(base, split='test')

per_image_metrics, all_keypoint_metrics, total_inference_time_sec = evaluate_SAM(
    sam_model,
    test_dataset,
    device,
    thresholds,
    use_windowed_softargmax,
    K=7,
    temperature=0.1,
    image_size = 512
)

save_results(per_image_metrics, all_keypoint_metrics, results_dir, total_inference_time_sec, thresholds)