import torch
from SPair71k.devkit.SPairDataset import SPairDataset
from evaluate import evaluate, save_results
from models.dinov2.dinov2.models.vision_transformer import vit_base
import os
from datetime import datetime
import json

# Configuration
base = '../Spair71k'
pair_ann_path = f'{base}/PairAnnotation'
layout_path = f'{base}/Layout'
image_path = f'{base}/JPEGImages'
dataset_size = 'large'
pck_alpha = 0.1  # mock, not used
thresholds = [0.05, 0.1, 0.2]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print()

# Load validation dataset
print("Loading validation dataset...")
val_dataset = SPairDataset(
    pair_ann_path,
    layout_path,
    image_path,
    dataset_size,
    pck_alpha,
    datatype='val'  # Changed to validation set
)
print(f"Validation samples: {len(val_dataset)}")
print()

# Models to evaluate
baseM = 'results_dinov2/ts_20251227_011648'
to_eval = [f"10temp_{i}_blocks_1epoch.pth" for i in range(1,4+1)]

# Results container for comparison
comparison_results = {}

# Evaluate each model
for model_name in to_eval:
    print("=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)
    print()

    # Create results_SPair71K directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.replace("10temp_", "").replace("1epoch.pth", "")
    results_dir = f'./blocks_comparison/validation_{model_short_name}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    print()

    # Load model
    print(f"Loading model: {model_name}")
    model = vit_base(
        img_size=(518, 518),
        patch_size=14,
        num_register_tokens=0,
        block_chunks=0,
        init_values=1.0,
    )

    model_path = f"./{baseM}/{model_name}"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Skipping this model...")
        print()
        continue

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    print()

    # Evaluate
    print("Starting evaluation on validation set...")
    per_image_metrics, all_keypoint_metrics, total_inference_time_sec = evaluate(
        model,
        val_dataset,
        device,
        thresholds,
        #early_stop=True
    )

    # Save results_SPair71K
    save_results(
        per_image_metrics,
        all_keypoint_metrics,
        results_dir,
        total_inference_time_sec,
        thresholds
    )
    import numpy as np

    # Calculate means from per_image_metrics
    pck_005_values = [m['pck_scores'][0.05] for m in per_image_metrics]
    pck_010_values = [m['pck_scores'][0.1] for m in per_image_metrics]
    pck_020_values = [m['pck_scores'][0.2] for m in per_image_metrics]

    # Store summary for comparison
    comparison_results[model_short_name] = {
        'pck@0.05': np.mean(pck_005_values),
        'pck@0.10': np.mean(pck_010_values),
        'pck@0.20': np.mean(pck_020_values),
        'inference_time': total_inference_time_sec,
        'results_dir': results_dir
    }

    print(f"✓ Evaluation completed for {model_name}")
    print()

# Print comparison summary
print()
print("=" * 80)
print("COMPARISON SUMMARY (Validation Set)")
print("=" * 80)
print()

print(f"{'Model':<20} {'PCK@0.05':<15} {'PCK@0.10':<15} {'PCK@0.20':<15} {'Time (s)':<12}")
print("-" * 80)

for model_name, results in comparison_results.items():
    print(
        f"{model_name:<20} {results['pck@0.05']:<15.2f} {results['pck@0.10']:<15.2f} {results['pck@0.20']:<15.2f} {results['inference_time']:<12.1f}")

print()

# Calculate differences
if len(comparison_results) == 2:
    models = list(comparison_results.keys())
    model1, model2 = models[0], models[1]

    print("=" * 80)
    print(f"DIFFERENCE: {model2} vs {model1}")
    print("=" * 80)
    print()

    for metric in ['pck@0.05', 'pck@0.10', 'pck@0.20']:
        diff = comparison_results[model2][metric] - comparison_results[model1][metric]
        pct_change = (diff / comparison_results[model1][metric]) * 100

        if diff > 0:
            print(f"{metric}: +{diff:.2f}% (+{pct_change:.1f}% relative) ✓ {model2} is better")
        elif diff < 0:
            print(f"{metric}: {diff:.2f}% ({pct_change:.1f}% relative) ✓ {model1} is better")
        else:
            print(f"{metric}: No difference")

    print()

# Save comparison to JSON
comparison_file = f'blocks_comparison/validation_comparison_{timestamp}.json'

with open(comparison_file, 'w') as f:
    json.dump(comparison_results, f, indent=2)


print(f"✓ Comparison results saved to: {comparison_file}")
print()
print("=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
