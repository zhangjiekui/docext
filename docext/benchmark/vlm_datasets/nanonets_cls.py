"""
This file contains code to convert the nanonets/Nanonets-Cls-Full dataset
into Nanonets IDP format. This is a document classification dataset. The dataset
contains single page and multi-page documents.

The dataset can be downloaded from:
https://huggingface.co/datasets/nanonets/Nanonets-Cls-Full
"""
from __future__ import annotations

import os

from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import Classification
from docext.benchmark.vlm_datasets.ds import ExtractionType


class NanonetsCls(BenchmarkDataset):
    name = "nanonets_cls"
    task = "CLASSIFICATION"

    def __init__(
        self,
        hf_name: str = "nanonets/Nanonets-Cls-Full",
        test_split: str = "test",
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        data: list[BenchmarkData] = self._load_data(
            hf_name, test_split, max_samples, cache_dir
        )
        super().__init__(self.name, data, cache_dir)

    def sample_class_wise_max_samples(
        self, dataset: Dataset, max_samples: int, class_labels: list[str]
    ):
        if max_samples is None or max_samples <= 0:
            return dataset
        sampled_data = []
        for class_label in class_labels:
            class_ids = [
                i for i in range(len(dataset)) if dataset[i]["label"] == class_label
            ]
            sorted_class_ids = sorted(class_ids)
            if len(class_ids) > max_samples:
                sampled_data.extend(dataset.select(sorted_class_ids[:max_samples]))
            else:
                sampled_data.extend(dataset.select(sorted_class_ids))
        return sampled_data

    def _load_data(
        self,
        hf_name: str,
        test_split: str,
        max_samples: int | None,
        cache_dir: str = "./docext_benchmark_cache",
    ):
        test_data = load_dataset(hf_name, split=test_split)
        class_labels = sorted(list(set(test_data["label"])))
        test_data = (
            self.sample_class_wise_max_samples(test_data, max_samples, class_labels)
            if max_samples is not None
            else test_data
        )

        data = []
        for i in tqdm(
            range(len(test_data)), desc=f"{self.name}: Converting data", leave=False
        ):
            data_point = test_data[i]
            images, label = (
                [self.bytes_to_image(image) for image in data_point["image"]],
                data_point["label"],
            )
            # save the image
            image_paths = []
            for j, image in enumerate(images):
                image_path = os.path.join(cache_dir, f"{i}_{j}.png")
                image = self.resize_image(image)
                image.save(image_path)
                image_paths.append(image_path)

            data.append(
                BenchmarkData(
                    image_paths=image_paths,
                    extraction_type=ExtractionType.CLASSIFICATION,
                    classification=Classification(doc_type=label, labels=class_labels),
                ),
            )
        return data


if __name__ == "__main__":
    dataset = NanonetsCls(max_samples=10)
