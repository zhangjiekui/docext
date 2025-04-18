"""
Start point for running the Nanonets IDP benchmark.


Checkout Nanonets for automating information extraction
from documents (like invoices, receipts, purchase orders, bills, etc) and automate workflows: https://nanonets.com/

Author: Souvik Mandal
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from typing import Dict
from typing import List

import json_repair
import pandas as pd
from litellm import completion
from loguru import logger
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from tqdm import tqdm

from docext.benchmark.metrics.kie import get_kie_metrics
from docext.benchmark.metrics.ocr import get_ocr_metrics
from docext.benchmark.metrics.vqa import get_vqa_extact_match_metrics
from docext.benchmark.tasks import get_datasets
from docext.benchmark.tasks import get_KIE_messages
from docext.benchmark.tasks import get_OCR_messages
from docext.benchmark.tasks import get_VQA_messages
from docext.benchmark.utils import load_yaml
from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import PredField
from docext.benchmark.vlm_datasets.ds import Prediction
from docext.benchmark.vlm_datasets.ds import VQA


class NanonetsIDPBenchmark:
    def __init__(
        self,
        benchmark_config_path: str,
    ):
        self.benchmark_config = load_yaml(benchmark_config_path)
        self._validate_benchmark_config(self.benchmark_config)

        # create the datasets
        self.datasets = self._get_datasets()

        for dataset in self.datasets:
            logger.info(f"Dataset {dataset.name} has {len(dataset.data)} samples")

        # create the models
        self.models = self.benchmark_config["models"]
        self.models = {model: self.benchmark_config[model] for model in self.models}

        # create the templates
        self.templates = {
            "KIE": self.benchmark_config["KIE_default_template"],
            "OCR": self.benchmark_config["OCR_default_template"],
            "VQA": self.benchmark_config["VQA_default_template"],
        }

        # run the benchmark, Note we cache each query. incase something fails, we can resume from the same point
        self.cache_dir = self.benchmark_config.get(
            "cache_dir",
            "./docext_benchmark_cache",
        )
        self.cache_dir = os.path.join(self.cache_dir, "prediction_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_datasets(self):
        datasets = get_datasets(
            self.benchmark_config["tasks"],
            self.benchmark_config["datasets"],
        )

        init_datasets = []

        for dataset in datasets:
            if dataset.name == "docile":
                init_datasets.append(
                    dataset(
                        annot_path=self.benchmark_config["docile"]["annot_path"],
                        annotations_root=self.benchmark_config["docile"][
                            "annotations_root"
                        ],
                        pdf_root=self.benchmark_config["docile"]["pdf_root"],
                        max_samples=self.benchmark_config.get(
                            "max_samples_per_dataset",
                            None,
                        ),
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            elif dataset.name == "handwritten_forms":
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["handwritten_forms"]["hf_name"],
                        test_split=self.benchmark_config["handwritten_forms"][
                            "test_split"
                        ],
                        max_samples=self.benchmark_config.get(
                            "max_samples_per_dataset",
                            None,
                        ),
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )

            elif dataset.name == "ocr_handwriting":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["ocr_handwriting"].get("max_samples", 1000),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["ocr_handwriting"]["hf_name"],
                        test_split=self.benchmark_config["ocr_handwriting"][
                            "test_split"
                        ],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            elif dataset.name == "digital_ocr_diacritics":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["digital_ocr_diacritics"].get(
                        "max_samples",
                        1000,
                    ),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["digital_ocr_diacritics"][
                            "hf_name"
                        ],
                        test_split=self.benchmark_config["digital_ocr_diacritics"][
                            "test_split"
                        ],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            elif dataset.name == "chartqa":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["chartqa"].get("max_samples", 1000),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["chartqa"]["hf_name"],
                        test_split=self.benchmark_config["chartqa"]["test_split"],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            else:
                raise ValueError(f"Dataset {dataset.name} is not supported.")
        return init_datasets

    def run_benchmark(self):
        all_scores = {}

        for dataset in tqdm(self.datasets):
            all_scores[dataset.name] = {}
            for model_name in self.models:
                benchmark_scores = self._run_single_model_single_dataset(
                    dataset,
                    model_name,
                    self.models[model_name],
                )
                all_scores[dataset.name][model_name] = benchmark_scores

        df = pd.DataFrame(all_scores)
        logger.info("\n" + df.to_string())
        return all_scores

    def _run_single_model_single_dataset(
        self,
        dataset: BenchmarkDataset,
        model_name: str,
        model_config: dict[str, Any],
    ):
        template = self.templates[dataset.task]
        if model_config.get("template", {}).get(dataset.task, None) is not None:
            # override the default template if provided in the model config
            template = model_config["template"][dataset.task]

        pred_with_gt = []

        for data in tqdm(
            dataset.data,
            desc=f"Running benchmark for {model_name} on {dataset.name}",
            leave=False,
        ):
            messages = self._get_messages(data, template, dataset.task)

            # check if the response is cached
            hash_messages = hashlib.sha256(str(messages).encode()).hexdigest()
            cache_file = os.path.join(
                self.cache_dir,
                f"{model_name.replace('/', '_')}_{hash_messages}.json",
            )

            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    response = json.load(f)
            else:
                # get the response from the model and cache it if it is not cached
                response = self._get_response(messages, model_name, model_config)

            # parse the response
            parsed_response = self._parse_response(response, dataset.task)
            if dataset.task == "KIE":
                pred_with_gt.append(
                    Prediction(
                        gt=data,
                        pred=BenchmarkData(
                            image_paths=data.image_paths,
                            extraction_type=data.extraction_type,
                            fields=[
                                PredField(
                                    label=label,
                                    value=value
                                    if isinstance(value, str)
                                    else ("" if value is None else json.dumps(value)),
                                    confidence=-1.0,
                                )
                                for label, value in parsed_response.items()
                            ],
                        ),
                    ),
                )
            elif dataset.task == "OCR":
                pred_with_gt.append(
                    Prediction(
                        gt=data,
                        pred=BenchmarkData(
                            image_paths=data.image_paths,
                            extraction_type=data.extraction_type,
                            ocr_text=parsed_response,
                        ),
                    ),
                )
            elif dataset.task == "VQA":
                pred_with_gt.append(
                    Prediction(
                        gt=data,
                        pred=BenchmarkData(
                            image_paths=data.image_paths,
                            extraction_type=data.extraction_type,
                            vqa=VQA(
                                question=data.vqa.question
                                if data.vqa is not None
                                else "",
                                answer=parsed_response,
                            ),
                        ),
                    ),
                )

        if dataset.task == "KIE":
            return get_kie_metrics(pred_with_gt)
        elif dataset.task == "OCR":
            return get_ocr_metrics(pred_with_gt)
        elif dataset.task == "VQA":
            return get_vqa_extact_match_metrics(pred_with_gt)
        else:
            raise ValueError(f"Task {dataset.task} is not supported.")

    def _get_messages(self, data: BenchmarkData, template: dict[str, Any], task: str):
        """
        Get the OpenAI-compatible messages for the given task and data.
        """
        if task == "KIE":
            return get_KIE_messages(data, template)
        elif task == "OCR":
            return get_OCR_messages(data, template)
        elif task == "VQA":
            return get_VQA_messages(data, template)
        else:
            raise ValueError(f"Task {task} is not supported.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=20, min=20, max=600),
    )
    def _get_response(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        model_config: dict[str, Any],
    ):

        response = completion(
            model=model_name,
            messages=messages,
            max_tokens=model_config.get("max_tokens", None),
            max_completion_tokens=model_config.get("max_completion_tokens", None),
            temperature=model_config.get("temperature", 0.0),
            # max_retries=3,
        )
        response = response.json()
        self._cache_response(messages, model_name, response)
        return response

    def _parse_response(self, response: dict, task: str):
        if task == "OCR" or task == "VQA":
            # OCR and VQA task returns a string
            answer = (
                response["choices"][0]["message"]["content"]
                if len(response["choices"]) > 0
                and response["choices"][0]["message"]["content"]
                else ""
            )
            return answer.strip() if answer else ""
        parsed_json = json_repair.repair_json(
            response["choices"][0]["message"]["content"]
            if len(response["choices"]) > 0
            and response["choices"][0]["message"]["content"]
            else "{}",
            ensure_ascii=False,
            return_objects=True,
        )
        if isinstance(parsed_json, list):  # TODO: can we handle this better?
            # merge all the keys into a single dict
            merged_dict = {}
            for item in parsed_json:
                merged_dict.update(item)
            return merged_dict

        if parsed_json == "":
            return {}  # parsing failed
        return parsed_json

    def _cache_response(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        response: str,
    ):
        """
        Cache the response for the given messages and model name.
        """
        hash_messages = hashlib.sha256(str(messages).encode()).hexdigest()
        cache_file = os.path.join(
            self.cache_dir,
            f"{model_name.replace('/', '_')}_{hash_messages}.json",
        )
        with open(cache_file, "w") as f:
            json.dump(response, f)

    def _validate_benchmark_config(self, benchmark_config: dict):
        # validate tasks
        assert "tasks" in benchmark_config, "tasks must be in the benchmark config"
        assert len(benchmark_config["tasks"]) > 0, "tasks must be non-empty"

        # validate dataset configs
        if "max_samples_per_dataset" in benchmark_config:
            assert (
                benchmark_config["max_samples_per_dataset"] > 0
            ), "max_samples_per_dataset must be a positive integer or `null` to consider all samples"

        if (
            "datasets" in benchmark_config
            and benchmark_config["datasets"] is not None
            and len(benchmark_config["datasets"]) > 0
        ):
            for dataset in benchmark_config["datasets"]:
                assert (
                    dataset in benchmark_config
                ), f"{dataset} config must be in the benchmark config"

        # validate default templates
        assert (
            "KIE_default_template" in benchmark_config
        ), "KIE_default_template must be in the benchmark config"
        assert (
            "system_prompt" in benchmark_config["KIE_default_template"]
        ), "system_prompt must be in the KIE_default_template"
        assert (
            "user_prompt" in benchmark_config["KIE_default_template"]
        ), "user_prompt must be in the KIE_default_template"
        assert (
            "document_page_seperator" in benchmark_config["KIE_default_template"]
        ), "document_page_seperator must be in the KIE_default_template"

        # validate models
        assert "models" in benchmark_config, "models must be in the benchmark config"
        assert len(benchmark_config["models"]) > 0, "models must be non-empty"
        for model in benchmark_config["models"]:
            assert (
                model in benchmark_config
            ), f"{model} config must be in the benchmark config"


if __name__ == "__main__":
    benchmark = NanonetsIDPBenchmark(
        benchmark_config_path="/home/paperspace/projects/docext/configs/benchmark.yaml",
    )
    benchmark.run_benchmark()
    print(benchmark.datasets)
