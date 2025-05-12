"""
Tasks that are supported by the Nanonets IDP benchmark.

Currently following tasks are supported:
1. KIE: Key Information Extraction
2. VQA: Visual Question Answering
3. OCR: Optical Character Recognition
4. Classification: Document Classification
5. LongDocBench: Long Document key information extraction

We plan to add more tasks in the future. If you are interested in adding a new task,
please refer to the README for instructions.
"""
from __future__ import annotations

from typing import Any

from docext.benchmark.utils import encode_image
from docext.benchmark.vlm_datasets.chartqa import ChartQA
from docext.benchmark.vlm_datasets.checkbox import DeathSe43_44_checkbox
from docext.benchmark.vlm_datasets.docile import Docile
from docext.benchmark.vlm_datasets.docvqa import DocVQA
from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.longdocbench import NanonetsLongDocBench
from docext.benchmark.vlm_datasets.nanonets_cls import NanonetsCls
from docext.benchmark.vlm_datasets.nanonets_kie import NanonetsKIE
from docext.benchmark.vlm_datasets.nanonets_tablebench import (
    NanonetsTableBenchLongDenseStructuredTable,
)
from docext.benchmark.vlm_datasets.nanonets_tablebench import (
    NanonetsTableBenchLongSparseStructuredTable,
)
from docext.benchmark.vlm_datasets.nanonets_tablebench import (
    NanonetsTableBenchLongSparseUntructuredTable,
)
from docext.benchmark.vlm_datasets.nanonets_tablebench import (
    NanonetsTableBenchSmallDenseStructuredTable,
)
from docext.benchmark.vlm_datasets.nanonets_tablebench import (
    NanonetsTableBenchSmallSparseStructuredTable,
)
from docext.benchmark.vlm_datasets.nanonets_tablebench import (
    NanonetsTableBenchSmallSparseUntructuredTable,
)
from docext.benchmark.vlm_datasets.ocr_dia import OCRDiacritics
from docext.benchmark.vlm_datasets.ocr_hw import OCRHandwritingHAT2023
from docext.benchmark.vlm_datasets.ocr_hw import OCRHandwritingRotated

KIE_DATASETS = [NanonetsKIE, Docile, DeathSe43_44_checkbox]

OCR_DATASETS = [
    OCRHandwritingHAT2023,
    OCRHandwritingRotated,
    OCRDiacritics,
]

VQA_DATASETS = [
    ChartQA,
    DocVQA,
    NanonetsLongDocBench,
]

CLASSIFICATION_DATASETS = [
    NanonetsCls,
]

TABLE_DATASETS = [
    NanonetsTableBenchSmallSparseStructuredTable,
    NanonetsTableBenchSmallDenseStructuredTable,
    NanonetsTableBenchSmallSparseUntructuredTable,
    NanonetsTableBenchLongDenseStructuredTable,
    NanonetsTableBenchLongSparseStructuredTable,
    NanonetsTableBenchLongSparseUntructuredTable,
]

TASKS2DATASETS = {
    "KIE": KIE_DATASETS,
    "OCR": OCR_DATASETS,
    "VQA": VQA_DATASETS,
    "CLASSIFICATION": CLASSIFICATION_DATASETS,
    "TABLE": TABLE_DATASETS,
}


def get_datasets(
    tasks: list[str],
    datasets: list[str] | None = None,
) -> list[BenchmarkDataset]:
    all_datasets: list[BenchmarkDataset] = []
    for task in tasks:
        all_datasets.extend(TASKS2DATASETS[task])  # type: ignore
    if datasets is not None:
        all_datasets = [d for d in all_datasets if d.name in datasets]
    return all_datasets


def get_image_encoding_type(image_path: str) -> str:
    if image_path.endswith(".png"):
        return "data:image/png;base64"
    elif image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        return "data:image/jpeg;base64"
    else:
        raise ValueError(f"Unsupported image format: {image_path}")


def get_TABLE_messages(data: BenchmarkData, template: dict[str, Any]):
    system_prompt = template["system_prompt"]
    document_page_seperator = template["document_page_seperator"]
    image_paths = data.image_paths
    columns = data.tables[0].columns if data.tables is not None else []
    assert len(columns) > 0, "No columns found in the data"

    output_format = [{col: "" for col in columns}]
    user_prompt = template["user_prompt"].format(
        columns=columns, output_format=output_format
    )

    image_user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_page_seperator.format(page_number=i + 1),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{get_image_encoding_type(filepath)},{encode_image(filepath)}",
                    },
                },
            ],
        }
        for i, filepath in enumerate(image_paths)
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *image_user_messages,
        {"role": "user", "content": user_prompt},
    ]
    return messages


def get_CLASSIFICATION_messages(data: BenchmarkData, template: dict[str, Any]):
    image_paths = data.image_paths
    labels = data.classification.labels if data.classification is not None else []
    assert len(labels) > 0, "No labels found in the data"
    system_prompt = template["system_prompt"].format(labels=labels)
    user_prompt = template["user_prompt"].format(labels=labels)
    document_page_seperator = template["document_page_seperator"]
    image_user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_page_seperator.format(page_number=i + 1),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{get_image_encoding_type(filepath)},{encode_image(filepath)}",
                    },
                },
            ],
        }
        for i, filepath in enumerate(image_paths)
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *image_user_messages,
        {"role": "user", "content": user_prompt},
    ]
    return messages


def get_VQA_messages(data: BenchmarkData, template: dict[str, Any]):
    system_prompt = template["system_prompt"]
    document_page_seperator = template["document_page_seperator"]
    image_paths = data.image_paths
    question = data.vqa.question if data.vqa is not None else ""
    assert question != "", "Question is empty"
    user_prompt = template["user_prompt"].format(
        question=question,
    )

    image_user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_page_seperator.format(page_number=i + 1),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{get_image_encoding_type(filepath)},{encode_image(filepath)}",
                    },
                },
            ],
        }
        for i, filepath in enumerate(image_paths)
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *image_user_messages,
        {"role": "user", "content": user_prompt},
    ]
    return messages


def get_OCR_messages(data: BenchmarkData, template: dict[str, Any]):
    system_prompt = template["system_prompt"]
    user_prompt = template["user_prompt"]
    image_paths = data.image_paths
    assert len(image_paths) == 1, "OCR task supports only single image"
    image_path = image_paths[0]
    image_user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{get_image_encoding_type(image_path)},{encode_image(image_path)}",
                    },
                },
            ],
        },
    ]
    return [
        {"role": "system", "content": system_prompt},
        *image_user_messages,
        {"role": "user", "content": user_prompt},
    ]


def get_KIE_messages(data: BenchmarkData, template: dict[str, Any]):
    system_prompt = template["system_prompt"]
    document_page_seperator = template["document_page_seperator"]
    image_paths = data.image_paths
    fields = data.fields or []
    field_names = [field.label for field in fields if field is not None]
    assert len(field_names) > 0, "No fields found in the data"
    output_format = {field: ".." for field in field_names}
    user_prompt = template["user_prompt"].format(
        fields=field_names,
        output_format=output_format,
    )

    image_user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_page_seperator.format(page_number=i + 1),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{get_image_encoding_type(filepath)},{encode_image(filepath)}",
                    },
                },
            ],
        }
        for i, filepath in enumerate(image_paths)
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *image_user_messages,
        {"role": "user", "content": user_prompt},
    ]
    return messages


def change_system_prompt(messages: list[dict[str, Any]], model_name: str):
    if "gemma-3-27b-it" in model_name:
        messages[0] = {"role": "user", "content": messages[0]["content"]}
    return messages
