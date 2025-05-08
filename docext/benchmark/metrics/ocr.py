from __future__ import annotations

from typing import List

from Levenshtein import distance as edit_distance

from docext.benchmark.vlm_datasets.ds import Prediction


def get_ocr_metrics(pred_with_gt: list[Prediction]):
    edit_distances = []
    for prediction in pred_with_gt:
        gt = prediction.gt
        pred = prediction.pred
        gt_ocr_text = gt.ocr_text if gt is not None and gt.ocr_text is not None else ""
        pred_ocr_text = (
            pred.ocr_text if pred is not None and pred.ocr_text is not None else ""
        )

        dist = edit_distance(pred_ocr_text, gt_ocr_text)
        max_len = max(len(pred_ocr_text), len(gt_ocr_text))
        if max_len == 0:
            edit_distances.append(
                1.0,
            )  # if both strings are empty, we consider it as 100% correct
        else:
            edit_distances.append(1 - (dist / max_len))
    return sum(edit_distances) / len(edit_distances)
