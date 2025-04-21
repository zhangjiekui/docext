from __future__ import annotations

from docext.benchmark.vlm_datasets.ds import Prediction


def get_classification_metrics(pred_with_gt: list[Prediction]):
    exact_matches = []
    for prediction in pred_with_gt:
        gt = prediction.gt
        pred = prediction.pred
        gt_answer = str(
            gt.classification.doc_type
            if gt is not None and gt.classification is not None
            else ""
        )
        pred_answer = str(
            pred.classification.doc_type
            if pred is not None and pred.classification is not None
            else ""
        )
        exact_match = str(gt_answer) == str(
            pred_answer
        )  # we convert to string to handle numbers
        exact_matches.append(exact_match)

    return sum(exact_matches) / len(exact_matches)
