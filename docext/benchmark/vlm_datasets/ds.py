from __future__ import annotations

import io
import os
import random
from enum import Enum
from typing import Union

import pandas as pd
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from pydantic import ConfigDict
from tqdm import tqdm

from docext.benchmark.vlm_datasets.utils import convert_pdf2image


class ExtractionType(Enum):
    FIELD = "field"
    TABLE = "table"
    CLASSIFICATION = "classification"
    OCR = "ocr"
    VQA = "vqa"


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Field(BaseModel):
    label: str
    value: str | list[str]
    description: str | None = None
    bbox: BBox | None = None
    page_path: str | None = (
        None  # for multi-page documents, this is the path to the page image
    )
    page_number: int | None = None  # for multi-page documents, this is the page number


class VQA(BaseModel):
    question: str
    answer: str | list[str]


class Classification(BaseModel):
    doc_type: str
    labels: list[str]


class PredField(Field):
    # predicted field
    confidence: float


class Table(BaseModel):
    table: pd.DataFrame
    columns: list[str | int]
    cell_boxes: list[BBox] | None = None
    name: str | None = None
    description: str | None = None
    bbox: BBox | None = None
    page_path: str | None = (
        None  # for multi-page documents, this is the path to the page image
    )
    page_number: int | None = None  # for multi-page documents, this is the page number

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BenchmarkData(BaseModel):
    image_paths: list[str]
    extraction_type: ExtractionType
    fields: list[Field | PredField] | None = None
    tables: list[Table] | None = None
    ocr_text: str | None = None
    classification: Classification | None = None
    vqa: VQA | None = None


class Prediction(BaseModel):
    gt: BenchmarkData | None = None
    pred: BenchmarkData | None = None

    def _get_pred_field_by_label(self, label: str):
        if self.pred is None or self.pred.fields is None:
            return ""
        for pred_field in self.pred.fields:
            if pred_field.label == label:
                return pred_field
        return ""


class BenchmarkDataset:
    task: str

    def __init__(
        self,
        name: str,
        data: list[BenchmarkData],
        cache_dir: str | None = None,
    ):
        """
        Each datapoint is expected to have following attributes:
        image_path: str
        extraction_type: enum of ["field", "classification"]
        fields: List[Field]
        classification: Optional[Classification] document type


        Field is expected to have following attributes:
        label: str
        value: str
        bbox: Optional[List[int]]
        """
        self.dataset_name = name
        self.data = data
        self.cache_dir = cache_dir

    def _get_cache_dir(self, dataset_name: str, cache_dir: str | None = None) -> str:
        if cache_dir is None:
            cache_dir = "./docext_benchmark_cache"
            cache_dir = os.path.join(cache_dir, dataset_name)
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache dir not provided, using {cache_dir}")
        else:
            cache_dir = os.path.join(cache_dir, dataset_name)
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def bytes_to_image(self, image_bytes: bytes):
        return Image.open(io.BytesIO(image_bytes))

    def __getitem__(self, index: int):
        return self.data[index]

    @property
    def name(self):
        return self.dataset_name

    def _convert_pdf_to_images(self, pdf_paths: list[str], cache_dir: str):
        pdf2image_mapping = {}
        for pdf_path in tqdm(
            pdf_paths,
            desc=f"{self.name}: Converting pdfs to images",
            leave=False,
        ):
            outdir = os.path.join(cache_dir, "pdf_images")
            os.makedirs(outdir, exist_ok=True)
            save_paths = convert_pdf2image(pdf_path, outdir)
            pdf2image_mapping[pdf_path] = save_paths
        return pdf2image_mapping

    def vis_random_sample(self):
        import matplotlib.pyplot as plt

        sample = random.choice(self.data)
        image_path = sample.image_paths[0]
        image = plt.imread(image_path)
        if sample.extraction_type == ExtractionType.FIELD:
            fields = sample.fields
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            ax = plt.gca()  # get the current axes

            for field in fields:
                bbox = field.bbox
                label = field.label
                if field.page_number != 0:
                    continue
                # plot the bbox and the label text
                ax.add_patch(
                    plt.Rectangle(
                        (bbox.x1, bbox.y1),
                        bbox.x2 - bbox.x1,
                        bbox.y2 - bbox.y1,
                        fill=False,
                        color="red",
                        linewidth=1,
                    ),
                )
                plt.text(
                    bbox.x1,
                    bbox.y1,
                    label,
                    fontsize=10,
                    color="red",
                    ha="left",
                    va="top",
                )

            plt.show()

    @property
    def field_labels(self):
        all_fields = []
        for sample in self.data:
            for field in sample.fields:
                all_fields.append(field.label)
        return list(set(all_fields))

    def resize_image(self, image: Image.Image, max_size: int = 1024):
        width, height = image.size
        if width > max_size or height > max_size:
            image = image.resize((max_size, max_size))
        return image
