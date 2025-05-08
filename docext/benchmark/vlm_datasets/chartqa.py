"""
This file contains code to convert the HuggingFaceM4/ChartQA dataset
into Nanonets IDP format. This is a question answering dataset for charts and plots.

The dataset can be downloaded from:
https://huggingface.co/datasets/HuggingFaceM4/ChartQA
"""
from __future__ import annotations

import os

from datasets import load_dataset
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType
from docext.benchmark.vlm_datasets.ds import VQA


class ChartQA(BenchmarkDataset):
    name = "chartqa"
    task = "VQA"

    def __init__(
        self,
        hf_name: str = "HuggingFaceM4/ChartQA",
        test_split: str = "test",
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        data: list[BenchmarkData] = self._load_data(
            hf_name, test_split, max_samples, cache_dir
        )
        super().__init__(self.name, data, cache_dir)

    def _load_data(
        self,
        hf_name: str,
        test_split: str,
        max_samples: int | None,
        cache_dir: str = "./docext_benchmark_cache",
    ):
        test_data = load_dataset(hf_name, split=test_split)
        test_data = (
            test_data.select(range(max_samples))
            if max_samples and max_samples > 0
            else test_data
        )

        data = []
        for i in tqdm(
            range(len(test_data)), desc=f"{self.name}: Converting data", leave=False
        ):
            data_point = test_data[i]
            image, query, label = (
                data_point["image"],
                data_point["query"],
                data_point["label"],
            )
            label = label[0]  # there is only one label always in the dataset
            # save the image
            image_path = os.path.join(cache_dir, f"{i}.png")
            image.save(image_path)
            data.append(
                BenchmarkData(
                    image_paths=[image_path],
                    extraction_type=ExtractionType.VQA,
                    vqa=VQA(question=query, answer=label),
                    classification=None,
                ),
            )
        return data
