from __future__ import annotations

import numpy as np

from docext.benchmark.metrics.grits import grits_from_df
from docext.benchmark.vlm_datasets.ds import Prediction


def get_table_metrics(pred_with_gt: list[Prediction]):
    metrics_list = []
    for prediction in pred_with_gt:
        gt = prediction.gt
        pred = prediction.pred
        gt_answer = (
            gt.tables[0].table if gt is not None and gt.tables is not None else ""
        )
        pred_answer = (
            pred.tables[0].table if pred is not None and pred.tables is not None else ""
        )
        metrics = grits_from_df(gt_answer, pred_answer)
        metrics_list.append(metrics)
    return np.mean(metrics_list)
