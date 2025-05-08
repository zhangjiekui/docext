"""
This file contains code to convert the Rasi1610/DeathSe43_44_checkbox dataset
into Nanonets IDP format. This is a handwritten form for death certificate.

The dataset can be downloaded from:
https://huggingface.co/datasets/Rasi1610/DeathSe43_44_checkbox

We skip two fields when converting the data:
1. death: Lots of incorrect annotations for death date. Sometimes its just year other times its full date.
2. birth: Lots of incorrect annotations for birth date. Sometimes its just year other times its full date.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType
from docext.benchmark.vlm_datasets.ds import Field


FIELDS_DESCRIPTIONS = {
    "deceased_status": "Status of the deceased, can be one of MARRIED, NEVER MARRIED, WIDOWED, DIVORCED",
}


class DeathSe43_44_checkbox(BenchmarkDataset):
    name = "handwritten_forms"
    task = "KIE"

    def __init__(
        self,
        hf_name: str = "Rasi1610/DeathSe43_44_checkbox",
        test_split: str = "val",
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        data = self._load_data(hf_name, test_split, max_samples, cache_dir)
        super().__init__(self.name, data, cache_dir)

    def _get_kie_data(self, ground_truth: str) -> dict:
        gt_parse = json.loads(ground_truth)["gt_parse"]
        final_gt_answer = {}
        for key, answer in gt_parse.items():
            for answer_key, answer_value in answer.items():
                if key == "person":
                    if answer_key == "death":
                        continue  # lots of incorrect annotations for death date. Sometimes its just year other times its full date.
                    elif answer_key in ["State file #"]:
                        final_gt_answer[answer_key] = answer_value
                    elif answer_key == "county":
                        final_gt_answer["place_of_death_county"] = answer_value
                    elif answer_key == "city":
                        final_gt_answer["place_of_death_city"] = answer_value
                    elif answer_key == "name":
                        final_gt_answer["name_of_deceased"] = answer_value
                elif key == "person_data":
                    if answer_key == "Gender":
                        final_gt_answer["deceased_gender"] = answer_value
                    elif answer_key == "Race":
                        final_gt_answer["deceased_race"] = answer_value
                    elif answer_key == "status":
                        final_gt_answer["deceased_status"] = answer_value
                    elif answer_key == "birth_day":
                        continue  # lots of incorrect annotations for birth date. Sometimes its just year other times its full date.
                    elif answer_key == "Age":
                        final_gt_answer["deceased_age"] = answer_value
                    elif answer_key == "birth_place":
                        final_gt_answer["birth_place"] = answer_value
                elif key == "relation":
                    if answer_key == "Father":
                        final_gt_answer["father_name"] = answer_value
                    elif answer_key == "Mother":
                        final_gt_answer["mother_name"] = answer_value
        return final_gt_answer

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

        ## convert the data to the format of the BenchmarkDataset
        data = []
        for i in tqdm(
            range(len(test_data)),
            desc=f"{self.name}: Converting data",
            leave=False,
        ):
            data_point = test_data[i]
            image, ground_truth = data_point["image"], data_point["ground_truth"]
            gt_answer = self._get_kie_data(ground_truth)
            gt_answer_object = [
                Field(label=k, value=v, description=FIELDS_DESCRIPTIONS.get(k, None))
                for k, v in gt_answer.items()
            ]
            # save the image
            image_path = os.path.join(cache_dir, f"{i}.png")
            image.save(image_path)
            data.append(
                BenchmarkData(
                    image_paths=[image_path],
                    extraction_type=ExtractionType.FIELD,
                    fields=gt_answer_object,
                    classification=None,
                ),
            )
        return data
