"""
LongDocBench is a dataset for long document key information extraction.
We concatinate multiple documents together to form a long document.
Then we ask to extract information from one of the such documents.

Eg: Extract fields3, fields4 from the document which has field1=value1, field2=value2.

We put the same document in multiple places in the long document to make it more challenging.
"""
from __future__ import annotations

import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType
from docext.benchmark.vlm_datasets.ds import Field
from docext.benchmark.vlm_datasets.ds import VQA

FIELDS_DESCRIPTIONS = {
    "deceased_status": "Status of the deceased, can be one of MARRIED, NEVER MARRIED, WIDOWED, DIVORCED",
}


class NanonetsLongDocBench(BenchmarkDataset):
    name = "nanonets_longdocbench"
    task = "VQA"

    def __init__(
        self,
        hf_name: str = "Rasi1610/DeathSe43_44_checkbox",
        test_split: str = "test",
        additional_docs_count: int = 20,
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        self.additional_docs_count = additional_docs_count
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
        additional_docs = load_dataset(hf_name, split="train")
        if max_samples and max_samples > 0:
            max_samples = min(max_samples, len(test_data))
            test_data = test_data.select(range(max_samples))

        additional_docs_image_paths = []
        for i in tqdm(
            range(len(additional_docs)),
            desc=f"{self.name}: Converting additional docs",
            leave=False,
        ):
            data_point = additional_docs[i]
            image = data_point["image"]
            image_path = os.path.join(cache_dir, f"additional_docs_{i}.png")
            image = self.resize_image(image)
            image.save(image_path)
            additional_docs_image_paths.append(image_path)

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

            # select a random field to ask
            random.seed(i)
            field2ask = random.choice(list(gt_answer.keys()))
            field2ask_answer = gt_answer[field2ask]
            gt_answer_object = [
                Field(label=k, value=v, description=FIELDS_DESCRIPTIONS.get(k, None))
                for k, v in gt_answer.items()
                if k != field2ask
            ]
            random.seed(i)
            random_image_order = random.sample(
                additional_docs_image_paths, self.additional_docs_count
            )
            # save the image
            image_path = os.path.join(cache_dir, f"{i}.png")
            image = self.resize_image(image)
            image.save(image_path)

            # Create 4 different lists with test image inserted at different positions
            insertion_points = [30, 60]
            for pos in insertion_points:
                insert_idx = int(len(random_image_order) * pos / 100)
                new_list = random_image_order.copy()
                new_list.insert(insert_idx, image_path)

                question = f"Extract {field2ask} from the image which has the following information: {gt_answer_object}. Just return the answer. Do not include any other text."
                answer = f"{field2ask_answer}"
                data.append(
                    BenchmarkData(
                        image_paths=new_list,
                        extraction_type=ExtractionType.VQA,
                        vqa=VQA(question=question, answer=answer),
                        classification=None,
                    ),
                )
        return data
