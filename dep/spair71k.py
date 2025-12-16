import os
import glob
import json
from typing import Text

def iterate_over_pairs(dataset_dir: Text, split: Text = 'test'):
    """Yields the pairs of images and corresponding ground truth points.
    """
    assert split in ['trn', 'test', 'val']
    for annotation in glob.glob(os.path.join(dataset_dir, 'PairAnnotation', split, '*.json')):
        with open(annotation) as f:
            data = json.load(f)
        category = data['category']
        source_path = os.path.join(
            dataset_dir, 'JPEGImages', category, data['src_imname'])
        target_path = os.path.join(
            dataset_dir, 'JPEGImages', category, data['trg_imname'])
        target_points = data['trg_kps']
        source_points = data['src_kps']
        target_pose = data['trg_pose']
        source_pose = data['src_pose']
        target_bounding_box = data['trg_bndbox']
        source_bounding_box = data['src_bndbox']
        result = {
            'source_path': source_path,
            'target_path': target_path,
            'source_points': source_points,
            'target_points': target_points,
            'category': category,
            'source_pose': source_pose,
            'target_pose': target_pose,
            "mirror": data['mirror'],
            "viewpoint_variation": data['viewpoint_variation'],
            "target_bounding_box": target_bounding_box,
            "source_bounding_box": source_bounding_box
        }
        yield result


