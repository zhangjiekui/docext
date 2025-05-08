"""
This file contains code to convert the DataStudio/OCR_handwritting_HAT2023 dataset
into Nanonets IDP format. This is a handwritten OCR dataset.

The dataset can be downloaded from:
https://huggingface.co/datasets/DataStudio/OCR_handwritting_HAT2023
"""
from __future__ import annotations

import os
import random
from typing import Optional

from tqdm import tqdm

from datasets import load_dataset
from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType


class OCRHandwritingHAT2023(BenchmarkDataset):
    name = "ocr_handwriting"
    task = "OCR"

    def __init__(
        self,
        hf_name: str = "DataStudio/OCR_handwritting_HAT2023",
        test_split: str = "val",
        max_samples: int | None = None,
        cache_dir: str | None = None,
        rotation: bool = False,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        self.rotation = rotation
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
            image, ocr_text = data_point["image"], data_point["text"]

            if self.rotation:
                random.seed(i)
                small_angle = random.choice(range(-5, 5))
                image = image.rotate(small_angle, expand=True)

            # save the image
            image_path = os.path.join(cache_dir, f"{i}.png")
            image.save(image_path)
            data.append(
                BenchmarkData(
                    image_paths=[image_path],
                    extraction_type=ExtractionType.OCR,
                    ocr_text=ocr_text,
                    classification=None,
                ),
            )
        return data


class OCRHandwritingRotated(OCRHandwritingHAT2023):
    name = "ocr_handwriting_rotated"
    task = "OCR"
