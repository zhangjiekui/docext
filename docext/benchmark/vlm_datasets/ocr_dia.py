"""
This file contains code to convert the ademax/ocr_scan_vi_01 dataset
into Nanonets IDP format. This is a digital OCR dataset with diacritics and
other non-latin characters.

The dataset can be downloaded from:
https://huggingface.co/datasets/ademax/ocr_scan_vi_01
"""
from __future__ import annotations

from typing import Optional

from docext.benchmark.vlm_datasets.ocr_hw import OCRHandwritingHAT2023


class OCRDiacritics(OCRHandwritingHAT2023):
    name = "digital_ocr_diacritics"
    task = "OCR"

    def __init__(
        self,
        hf_name: str = "ademax/ocr_scan_vi_01",
        test_split: str = "test",
        max_samples: int | None = None,
        cache_dir: str | None = None,
        rotation: bool = False,
    ):
        super().__init__(
            hf_name=hf_name,
            test_split=test_split,
            max_samples=max_samples,
            cache_dir=cache_dir,
            rotation=rotation,
        )
