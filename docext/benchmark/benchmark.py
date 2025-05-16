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
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Any

import json_repair
import mdpd
import pandas as pd
from litellm import completion
from loguru import logger
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from tqdm import tqdm

from docext.benchmark.metrics.classification import get_classification_metrics
from docext.benchmark.metrics.kie import get_kie_metrics
from docext.benchmark.metrics.ocr import get_ocr_metrics
from docext.benchmark.metrics.tables import get_table_metrics
from docext.benchmark.metrics.vqa import get_vqa__metric_for_multiple_possible_answers
from docext.benchmark.metrics.vqa import get_vqa_metrics
from docext.benchmark.tasks import change_system_prompt
from docext.benchmark.tasks import get_CLASSIFICATION_messages
from docext.benchmark.tasks import get_datasets
from docext.benchmark.tasks import get_KIE_messages
from docext.benchmark.tasks import get_OCR_messages
from docext.benchmark.tasks import get_TABLE_messages
from docext.benchmark.tasks import get_VQA_messages
from docext.benchmark.tasks import TABLE_DATASETS
from docext.benchmark.utils import load_yaml
from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import Classification
from docext.benchmark.vlm_datasets.ds import PredField
from docext.benchmark.vlm_datasets.ds import Prediction
from docext.benchmark.vlm_datasets.ds import Table
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
            "CLASSIFICATION": self.benchmark_config["CLASSIFICATION_default_template"],
            "TABLE": self.benchmark_config["TABLE_default_template"],
        }

        # run the benchmark, Note we cache each query. incase something fails, we can resume from the same point
        self.cache_dir = self.benchmark_config.get(
            "cache_dir",
            "./docext_benchmark_cache",
        )
        self.cache_dir = os.path.join(self.cache_dir, "prediction_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.max_workers = self.benchmark_config.get("max_workers", 1)
        self.ignore_cache = self.benchmark_config.get("ignore_cache", False)

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
            elif dataset.name == "ocr_handwriting_rotated":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["ocr_handwriting_rotated"].get(
                        "max_samples", 1000
                    ),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["ocr_handwriting_rotated"][
                            "hf_name"
                        ],
                        test_split=self.benchmark_config["ocr_handwriting_rotated"][
                            "test_split"
                        ],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                        rotation=True,
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
            elif dataset.name == "docvqa":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["docvqa"].get("max_samples", 1000),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["docvqa"]["hf_name"],
                        test_split=self.benchmark_config["docvqa"]["test_split"],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )

            elif dataset.name == "nanonets_longdocbench":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["nanonets_longdocbench"].get(
                        "max_samples", 1000
                    ),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["nanonets_longdocbench"][
                            "hf_name"
                        ],
                        test_split=self.benchmark_config["nanonets_longdocbench"][
                            "test_split"
                        ],
                        additional_docs_count=self.benchmark_config[
                            "nanonets_longdocbench"
                        ]["additional_docs_count"],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            elif dataset.name == "nanonets_cls":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["nanonets_cls"].get("max_samples", 1000),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["nanonets_cls"]["hf_name"],
                        test_split=self.benchmark_config["nanonets_cls"]["test_split"],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            elif dataset.name == "nanonets_kie":
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config["nanonets_kie"].get("max_samples", 1000),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config["nanonets_kie"]["hf_name"],
                        test_split=self.benchmark_config["nanonets_kie"]["test_split"],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )

            elif dataset.name in [ds.name for ds in TABLE_DATASETS]:
                max_samples = self.benchmark_config.get("max_samples_per_dataset", None)
                max_samples = min(
                    max_samples,
                    self.benchmark_config[dataset.name].get("max_samples", 1000),
                )
                init_datasets.append(
                    dataset(
                        hf_name=self.benchmark_config[dataset.name]["hf_name"],
                        test_split=self.benchmark_config[dataset.name]["test_split"],
                        max_samples=max_samples,
                        cache_dir=self.benchmark_config.get("cache_dir", None),
                    ),
                )
            else:
                raise ValueError(f"Dataset {dataset.name} is not supported.")
        return init_datasets

    def run_benchmark(self):
        all_scores = {}
        all_costs = {}
        for dataset in tqdm(self.datasets):
            all_scores[dataset.name] = {}
            all_costs[dataset.name] = {}
            for model_name in self.models:
                benchmark_scores, avg_cost = self._run_single_model_single_dataset(
                    dataset,
                    model_name,
                    self.models[model_name],
                )
                all_scores[dataset.name][model_name] = benchmark_scores
                all_costs[dataset.name][model_name] = avg_cost
        df = pd.DataFrame(all_scores)
        df["average"] = df.mean(axis=1)
        df = df[["average"] + list(df.columns[:-1])]
        df = df.sort_values(by="average", ascending=False)
        logger.info("ACCURACY:\n" + df.to_string())

        ## save the df to a csv file
        df.to_csv("accuracy.csv", index=True)

        # for cost show the avg for each model
        df_cost = pd.DataFrame(all_costs)
        df_cost = df_cost.mean(axis=1)
        # sort the df_cost by the index
        df_cost = df_cost.sort_index()
        logger.info("COST:\n" + df_cost.to_string())
        df_cost.to_csv("cost.csv", index=True)
        return all_scores, all_costs

    def _get_prediction_cache_files(self, dataset: BenchmarkDataset, model_name: str):
        template = self.templates[dataset.task]
        model_config = self.models[model_name]
        if model_config.get("template", {}).get(dataset.task, None) is not None:
            # override the default template if provided in the model config
            template = model_config["template"][dataset.task]

        cache_files = []
        questions = []
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
                cache_files.append(cache_file)
                questions.append(messages[-1]["content"])
        return cache_files, questions

    def _process_item(
        self,
        data: BenchmarkData,
        template: dict[str, Any],
        task: str,
        model_name: str,
        model_config: dict[str, Any],
    ):
        messages = self._get_messages(data, template, task)
        messages = change_system_prompt(messages, model_name)
        hash_messages = hashlib.sha256(str(messages).encode()).hexdigest()
        cache_file = os.path.join(
            self.cache_dir,
            f"{model_name.replace('/', '_')}_{hash_messages}.json",
        )

        if os.path.exists(cache_file) and not self.ignore_cache:
            with open(cache_file) as f:
                response = json.load(f)
            if (
                len(response["choices"]) > 0
                # and response["choices"][0]["message"]["content"] not in [None, ""]
                and response["choices"][0]["finish_reason"] == "stop"
            ):
                return response
        # get the response from the model and cache it if it is not cached
        response = self._get_response(messages, model_name, model_config)
        # if response is None:
        #     return None
        # if len(response["choices"]) > 0 and response["choices"][0]["finish_reason"] == "stop":
        #     return response
        # else:
        #     breakpoint()
        return response

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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = list(
                tqdm(
                    executor.map(
                        self._process_item,
                        dataset.data,
                        repeat(template),
                        repeat(dataset.task),
                        repeat(model_name),
                        repeat(model_config),
                    ),
                    total=len(dataset.data),
                    desc=f"Running benchmark for {model_name} on {dataset.name}",
                    leave=False,
                )
            )

        responses = list(futures)
        old_num_responses = len(responses)
        # # save the image paths of the None responses.
        # for response, data in zip(responses, dataset.data):
        #     if response is None:
        #         print(data.image_paths)
        # dataset.data = [data for data, response in zip(dataset.data, responses) if response is not None]
        # responses = [response for response in responses if response is not None]
        # assert len(dataset.data) == len(responses)
        # if old_num_responses != len(responses):
        #     logger.info(f"Total responses: Before filtering: {old_num_responses}, after filtering: {len(responses)} for {model_name} on {dataset.name}")
        total_cost = 0
        for response, data in zip(responses, dataset.data):
            total_cost += (
                response["response_cost"]
                if response["response_cost"] is not None
                else 0
            )
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

            elif dataset.task == "CLASSIFICATION":
                pred_with_gt.append(
                    Prediction(
                        gt=data,
                        pred=BenchmarkData(
                            extraction_type=data.extraction_type,
                            image_paths=data.image_paths,
                            classification=Classification(
                                doc_type=parsed_response,
                                labels=data.classification.labels
                                if data.classification is not None
                                else [],
                            ),
                        ),
                    ),
                )
            elif dataset.task == "TABLE":
                parsed_response = (
                    [parsed_response]
                    if isinstance(parsed_response, pd.DataFrame)
                    else parsed_response
                )
                pred_with_gt.append(
                    Prediction(
                        gt=data,
                        pred=BenchmarkData(
                            image_paths=data.image_paths,
                            extraction_type=data.extraction_type,
                            tables=[
                                Table(
                                    table=table,
                                    columns=table.columns.tolist(),
                                )
                                for table in parsed_response
                            ],
                        ),
                    ),
                )

        avg_cost = total_cost / len(responses)
        if dataset.task == "KIE":
            return get_kie_metrics(pred_with_gt), avg_cost
        elif dataset.task == "OCR":
            return get_ocr_metrics(pred_with_gt), avg_cost
        elif dataset.task == "VQA":
            if dataset.name == "docvqa":
                return (
                    get_vqa__metric_for_multiple_possible_answers(pred_with_gt),
                    avg_cost,
                )
            else:
                return get_vqa_metrics(pred_with_gt), avg_cost
        elif dataset.task == "CLASSIFICATION":
            return get_classification_metrics(pred_with_gt), avg_cost
        elif dataset.task == "TABLE":
            return get_table_metrics(pred_with_gt), avg_cost
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
        elif task == "CLASSIFICATION":
            return get_CLASSIFICATION_messages(data, template)
        elif task == "TABLE":
            return get_TABLE_messages(data, template)
        else:
            raise ValueError(f"Task {task} is not supported.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=10, min=60, max=600),
    )
    def _get_response(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        model_config: dict[str, Any],
    ):
        # import litellm
        # litellm._turn_on_debug()
        # breakpoint()
        if model_name.startswith("hosted_vllm/"):
            assert (
                model_config.get("api_key", None) is not None
            ), "api_key must be provided for hosted_vllm models"
            assert (
                model_config.get("api_base", None) is not None
            ), "api_base must be provided for hosted_vllm models"

        response = completion(
            model=model_name,
            messages=messages,
            max_tokens=model_config.get("max_tokens", None),
            max_completion_tokens=model_config.get("max_completion_tokens", None),
            temperature=model_config.get("temperature", 0.0),
            reasoning_effort="low",
            drop_params=True,
            api_key=model_config.get("api_key", None)
            if model_name.startswith("hosted_vllm/")
            else None,
            api_base=model_config.get("api_base", None)
            if model_name.startswith("hosted_vllm/")
            else None,
        )
        response_cost = response._hidden_params["response_cost"]
        token_counter = (
            response._hidden_params["token_counter"]
            if "token_counter" in response._hidden_params
            else -1
        )
        response = response.json()
        response["response_cost"] = response_cost
        response["token_counter"] = token_counter

        self._cache_response(messages, model_name, response)
        return response

    def _parse_response(self, response: dict, task: str):
        if task == "OCR" or task == "VQA" or task == "CLASSIFICATION":
            # OCR and VQA, Classification task returns a string
            answer = (
                response["choices"][0]["message"]["content"]
                if len(response["choices"]) > 0
                and response["choices"][0]["message"]["content"]
                else ""
            )
            return answer.strip() if answer else ""
        elif task == "TABLE":
            response = response["choices"][0]["message"]["content"]

            # convert the parsed_json to a dataframe
            try:
                parsed_json = json_repair.repair_json(
                    response, ensure_ascii=False, return_objects=True
                )
                if isinstance(parsed_json[0], list):
                    df = pd.concat([pd.DataFrame(item) for item in parsed_json])
                else:
                    df = pd.DataFrame(parsed_json)
                return df
            except Exception as e:
                print(f"Error parsing table: {e}")
                return pd.DataFrame()

        parsed_json = json_repair.repair_json(
            response["choices"][0]["message"]["content"]
            if len(response["choices"]) > 0
            and response["choices"][0]["message"]["content"]
            else "{}",
            ensure_ascii=False,
            return_objects=True,
        )
        if isinstance(parsed_json, list):
            # merge all the keys into a single dict
            merged_dict = {}
            for item in parsed_json:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key not in merged_dict:
                            merged_dict[key] = value
                        else:
                            if isinstance(merged_dict[key], list):
                                merged_dict[key].append(value)
                            else:
                                merged_dict[key] = [merged_dict[key], value]
                # we ignore the other types of objects, the model should not return them
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
