from __future__ import annotations

from typing import List

from Levenshtein import distance as edit_distance

from docext.benchmark.vlm_datasets.ds import Prediction


def get_vqa_metrics(pred_with_gt: list[Prediction]):
    exact_matches = []
    for prediction in pred_with_gt:
        gt = prediction.gt
        pred = prediction.pred
        gt_answer = str(gt.vqa.answer if gt is not None and gt.vqa is not None else "")
        pred_answer = str(
            pred.vqa.answer if pred is not None and pred.vqa is not None else ""
        )
        # print(gt_answer, pred_answer)
        # exact_match = str(gt_answer) == str(
        #     pred_answer
        # )  # we convert to string to handle numbers
        # exact_matches.append(exact_match)
        dist = edit_distance(pred_answer, gt_answer)
        max_len = max(len(pred_answer), len(gt_answer))
        if max_len == 0:
            exact_matches.append(1.0)
        else:
            exact_matches.append(1 - (dist / max_len))

    return sum(exact_matches) / len(exact_matches)
