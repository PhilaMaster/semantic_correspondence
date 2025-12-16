import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os



def load_and_analyze(results_dir):
    """
    Load all_keypoint_metrics.csv and compute overall and per-category statistics.
    """

    print(f"Loading data from {results_dir}...")
    df_keypoints = pd.read_csv(f'{results_dir}/all_keypoint_metrics.csv')

    print(f"Loaded {len(df_keypoints)} keypoint measurements")
    print(f"Unique images: {df_keypoints['image_idx'].nunique()}")
    print(f"Categories: {df_keypoints['category'].unique()}")
    print(f"Thresholds: {df_keypoints['threshold'].unique()}")

    thresholds = sorted(df_keypoints['threshold'].unique())

    # ===================== OVERALL RESULTS =====================
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    overall_stats = {}

    for threshold in thresholds:
        df_thresh = df_keypoints[df_keypoints['threshold'] == threshold]

        # Compute per-image PCK (average over all keypoints per image)
        per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean() * 100

        mean_pck = float(per_image_pck.mean())
        std_pck = float(per_image_pck.std())
        median_pck = float(per_image_pck.median())
        p25 = float(per_image_pck.quantile(0.25))
        p75 = float(per_image_pck.quantile(0.75))

        overall_stats[f"pck@{threshold:.2f}"] = {
            "mean": mean_pck,
            "std": std_pck,
            "median": median_pck,
            "p25": p25,
            "p75": p75,
        }

        print(f"PCK@{threshold:.2f}: "
              f"mean={mean_pck:.2f}%, std={std_pck:.2f}%, "
              f"median={median_pck:.2f}%, "
              f"p25={p25:.2f}%, p75={p75:.2f}%")

    with open(f'{results_dir}/overall_stats.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)

    # ===================== PER-CATEGORY RESULTS =====================
    print("\n" + "=" * 60)
    print("PER-CATEGORY RESULTS")
    print("=" * 60)

    per_category_stats = {}

    for category in sorted(df_keypoints['category'].unique()):
        print(f"\nCategory: {category}")
        per_category_stats[category] = {}

        df_cat = df_keypoints[df_keypoints['category'] == category]

        for threshold in thresholds:
            df_thresh = df_cat[df_cat['threshold'] == threshold]

            per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean() * 100

            mean_pck = float(per_image_pck.mean())
            std_pck = float(per_image_pck.std())
            median_pck = float(per_image_pck.median())

            per_category_stats[category][f"pck@{threshold:.2f}"] = {
                "mean": mean_pck,
                "std": std_pck,
                "median": median_pck,
            }

            print(f"  PCK@{threshold:.2f}: "
                  f"mean={mean_pck:.2f}%, std={std_pck:.2f}%, median={median_pck:.2f}%")

    with open(f'{results_dir}/per_category_stats.json', 'w') as f:
        json.dump(per_category_stats, f, indent=2)

    # category CSV
    category_rows = []
    for category in sorted(df_keypoints['category'].unique()):
        df_cat = df_keypoints[df_keypoints['category'] == category]
        for threshold in thresholds:
            df_thresh = df_cat[df_cat['threshold'] == threshold]
            per_image_pck = df_thresh.groupby('image_idx')['correct_at_threshold'].mean() * 100

            category_rows.append({
                'category': category,
                'threshold': threshold,
                'mean_pck': per_image_pck.mean(),
                'std_pck': per_image_pck.std(),
                'median_pck': per_image_pck.median(),
                'count': len(per_image_pck)
            })

    df_category = pd.DataFrame(category_rows)
    df_category.to_csv(f'{results_dir}/per_category_metrics.csv', index=False)

    # ===================== KEYPOINT ANALYSYS =====================
    print("\n" + "=" * 60)
    print("PER-KEYPOINT ANALYSIS")
    print("=" * 60)

    threshold_to_analyze = 0.10
    df_kp = df_keypoints[df_keypoints['threshold'] == threshold_to_analyze].copy()

    print(f"\n--- Global Keypoint Difficulty (PCK@{threshold_to_analyze}) ---\n")

    keypoint_stats = df_kp.groupby('keypoint_id').agg({
        'correct_at_threshold': ['mean', 'count'],
        'distance': ['mean', 'std']
    }).round(4)

    keypoint_stats.columns = ['accuracy', 'count', 'avg_distance', 'std_distance']
    keypoint_stats['accuracy_pct'] = keypoint_stats['accuracy'] * 100
    keypoint_stats = keypoint_stats.sort_values('accuracy_pct')

    print(keypoint_stats.to_string())

    print(f"\nMost Difficult Keypoints (lowest accuracy):")
    print(keypoint_stats.head(5)[['accuracy_pct', 'avg_distance', 'count']])

    print(f"\nEasiest Keypoints (highest accuracy):")
    print(keypoint_stats.tail(5)[['accuracy_pct', 'avg_distance', 'count']])

    keypoint_stats.to_csv(f'{results_dir}/keypoint_global_stats.csv')

    # ===================== CATEGORY AND KEYPOINT ANALYSIS =====================
    print(f"\n--- Per-Category Keypoint Analysis (PCK@{threshold_to_analyze}) ---\n")

    category_keypoint_stats = df_kp.groupby(['category', 'keypoint_id']).agg({
        'correct_at_threshold': ['mean', 'count'],
        'distance': ['mean', 'std']
    }).round(4)

    category_keypoint_stats.columns = ['accuracy', 'count', 'avg_distance', 'std_distance']
    category_keypoint_stats['accuracy_pct'] = category_keypoint_stats['accuracy'] * 100

    for category in sorted(df_kp['category'].unique()):
        print(f"\n=== Category: {category} ===")
        cat_stats = category_keypoint_stats.loc[category].sort_values('accuracy_pct')

        if len(cat_stats) > 0:
            print(f"\nKeypoint difficulty ranking:")
            print(cat_stats[['accuracy_pct', 'avg_distance', 'count']].to_string())

    category_keypoint_stats.to_csv(f'{results_dir}/keypoint_per_category_stats.csv')

    # ===================== VISUALIZZAZIONI =====================
    print("\n--- Generating Visualizations ---")

    #heatmap
    pivot_accuracy = df_kp.pivot_table(
        values='correct_at_threshold',
        index='keypoint_id',
        columns='category',
        aggfunc='mean'
    ) * 100

    pivot_count = df_kp.pivot_table(
        values='correct_at_threshold',
        index='keypoint_id',
        columns='category',
        aggfunc='count'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    #heatmap accuracy
    im1 = axes[0].imshow(pivot_accuracy.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0].set_title(f'Keypoint Accuracy per Category (PCK@{threshold_to_analyze})',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Category', fontsize=12)
    axes[0].set_ylabel('Keypoint ID', fontsize=12)
    axes[0].set_xticks(range(len(pivot_accuracy.columns)))
    axes[0].set_xticklabels(pivot_accuracy.columns, rotation=45, ha='right')
    axes[0].set_yticks(range(len(pivot_accuracy.index)))
    axes[0].set_yticklabels(pivot_accuracy.index)

    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Accuracy (%)', fontsize=12)

    for i in range(len(pivot_accuracy.index)):
        for j in range(len(pivot_accuracy.columns)):
            if not np.isnan(pivot_accuracy.values[i, j]):
                axes[0].text(j, i, f'{pivot_accuracy.values[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)

    # Heatmap conteggio
    im2 = axes[1].imshow(pivot_count.values, cmap='Blues', aspect='auto')
    axes[1].set_title(f'Number of Samples per Keypoint', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Category', fontsize=12)
    axes[1].set_ylabel('Keypoint ID', fontsize=12)
    axes[1].set_xticks(range(len(pivot_count.columns)))
    axes[1].set_xticklabels(pivot_count.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(pivot_count.index)))
    axes[1].set_yticklabels(pivot_count.index)

    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Sample Count', fontsize=12)

    for i in range(len(pivot_count.index)):
        for j in range(len(pivot_count.columns)):
            if not np.isnan(pivot_count.values[i, j]):
                axes[1].text(j, i, f'{int(pivot_count.values[i, j])}',
                             ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/keypoint_analysis_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved")

    # Box plot
    fig, ax = plt.subplots(figsize=(14, 6))

    keypoint_indices = sorted(df_kp['keypoint_id'].unique())
    distances_by_kp = [df_kp[df_kp['keypoint_id'] == kp]['distance'].values
                       for kp in keypoint_indices]

    bp = ax.boxplot(distances_by_kp, labels=keypoint_indices, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.axhline(y=threshold_to_analyze, color='r', linestyle='--',
               label=f'Threshold ({threshold_to_analyze})')
    ax.set_xlabel('Keypoint ID', fontsize=12)
    ax.set_ylabel('Normalized Distance', fontsize=12)
    ax.set_title('Error Distribution per Keypoint', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/keypoint_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Box plot saved")

    # ===================== SUMMARY REPORT =====================
    with open(f'{results_dir}/summary_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DINOV2 SPAIR-71K EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("OVERALL RESULTS\n")
        f.write("-" * 60 + "\n")
        for threshold in thresholds:
            stats = overall_stats[f"pck@{threshold:.2f}"]
            f.write(f"PCK@{threshold:.2f}: mean={stats['mean']:.2f}%, "
                    f"std={stats['std']:.2f}%, median={stats['median']:.2f}%\n")

        f.write("\n\nPER-CATEGORY RESULTS\n")
        f.write("-" * 60 + "\n")
        for category in sorted(per_category_stats.keys()):
            f.write(f"\n{category}:\n")
            for threshold in thresholds:
                stats = per_category_stats[category][f"pck@{threshold:.2f}"]
                f.write(f"  PCK@{threshold:.2f}: mean={stats['mean']:.2f}%, "
                        f"std={stats['std']:.2f}%\n")

        f.write("\n\nKEYPOINT ANALYSIS SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total keypoints analyzed: {len(df_kp)}\n")
        f.write(f"Unique keypoint IDs: {df_kp['keypoint_id'].nunique()}\n")
        f.write(f"Categories: {df_kp['category'].nunique()}\n")
        f.write(f"\nAccuracy range: {keypoint_stats['accuracy_pct'].min():.2f}% - "
                f"{keypoint_stats['accuracy_pct'].max():.2f}%\n")

    print(f"\n✓ Analysis complete! All results saved to: {results_dir}")
    return df_keypoints, overall_stats, per_category_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process keypoint metrics from all_keypoint_metrics.csv")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=".",
        help="Directory containing all_keypoint_metrics.csv"
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        exit(1)

    csv_path = os.path.join(results_dir, "all_keypoint_metrics.csv")
    if not os.path.isfile(csv_path):
        print(f"all_keypoint_metrics.csv file not found inside: {csv_path}")
        exit(1)

    load_and_analyze(results_dir)
