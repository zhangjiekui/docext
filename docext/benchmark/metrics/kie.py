from __future__ import annotations

from typing import List

from Levenshtein import distance as edit_distance
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import Prediction


def get_kie_metrics(predictions: list[Prediction]):
    """
    Get the metrics for the predictions.
    """
    edit_distances = []
    for pred in tqdm(predictions, desc="Computing KIE metrics", leave=False):
        gt_fields = pred.gt.fields
        for gt_field in gt_fields:
            pred_field = pred._get_pred_field_by_label(gt_field.label)
            if pred_field is None or pred_field == "":
                pred_value = ""
            else:
                pred_value = pred_field.value
            pred_value = str(pred_value)
            gt_value = str(gt_field.value)
            dist = edit_distance(pred_value, gt_value)
            max_len = max(len(pred_value), len(gt_value))
            if max_len == 0:
                edit_distances.append(1.0)
            else:
                edit_distances.append(1 - (dist / max_len))
    return sum(edit_distances) / len(edit_distances)
