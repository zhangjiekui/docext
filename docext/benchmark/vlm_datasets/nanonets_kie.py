"""
This file contains code to convert the `nanonets/key_information_extraction` dataset
into Nanonets IDP format. This is a key information extraction dataset. The dataset contains
receipts, and are annotated for following fields:
"date", "doc_no_receipt_no", "seller_address", "seller_gst_id", "seller_name", "seller_phone",
"total_amount", "total_tax"

The dataset can be downloaded from:
https://huggingface.co/datasets/nanonets/key_information_extraction
"""
from __future__ import annotations

import os

from tqdm import tqdm

from datasets import Dataset
from datasets import load_dataset
from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType
from docext.benchmark.vlm_datasets.ds import Field


class NanonetsKIE(BenchmarkDataset):
    name = "nanonets_kie"
    task = "KIE"

    def __init__(
        self,
        hf_name: str = "nanonets/key_information_extraction",
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
        if max_samples and max_samples > 0:
            max_samples = min(max_samples, len(test_data))
            test_data = test_data.select(range(max_samples))

        data = []
        for i in tqdm(
            range(len(test_data)), desc=f"{self.name}: Converting data", leave=False
        ):
            data_point = test_data[i]
            image, label = (
                self.bytes_to_image(data_point["image"]),
                data_point["annotations"],
            )
            # save the image
            image_path = os.path.join(cache_dir, f"{i}.png")
            image.save(image_path)

            data.append(
                BenchmarkData(
                    image_paths=[image_path],
                    extraction_type=ExtractionType.FIELD,
                    fields=[Field(label=k, value=v) for k, v in label.items()],
                ),
            )
        return data


if __name__ == "__main__":
    dataset = NanonetsKIE(max_samples=10)
