from __future__ import annotations

import os
from typing import Optional

from loguru import logger
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import BBox
from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType
from docext.benchmark.vlm_datasets.ds import Field
from docext.benchmark.vlm_datasets.utils import load_json


class Docile(BenchmarkDataset):
    name = "docile"
    task = "KIE"

    def __init__(
        self,
        annot_path: str,
        annotations_root: str | None = None,
        pdf_root: str | None = None,
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        ## convert the data to the format of the BenchmarkDataset
        data = self._convert_data(
            annot_path,
            annotations_root,
            pdf_root,
            max_samples,
            cache_dir,
        )
        super().__init__(self.name, data, cache_dir)

    def _convert_data(
        self,
        annot_path: str,
        annotations_root: str | None = None,
        pdf_root: str | None = None,
        max_samples: int | None = None,
        cache_dir: str = "./docext_benchmark_cache",
    ):
        data_ids = load_json(annot_path)

        if max_samples is not None and max_samples > 0:
            data_ids = data_ids[:max_samples]

        annotations_root = annotations_root or os.path.join(
            os.path.dirname(annot_path),
            "annotations",
        )
        annotation_paths = [
            os.path.join(annotations_root, f"{id}.json") for id in data_ids
        ]
        annotation_paths = [p for p in annotation_paths if os.path.exists(p)]

        # logger.debug(f"Found {len(annotation_paths)} annotations")

        pdf_root = pdf_root or os.path.join(os.path.dirname(annot_path), "pdfs")
        pdf_paths = [os.path.join(pdf_root, f"{id}.pdf") for id in data_ids]
        pdf_paths = [p for p in pdf_paths if os.path.exists(p)]
        # logger.debug(f"Found {len(pdf_paths)} pdfs")

        assert len(pdf_paths) == len(annotation_paths)
        # convert the pdf data to images
        pdf2image_mapping = self._convert_pdf_to_images(pdf_paths, cache_dir)

        data = []
        for pdf_path, annotation_path in tqdm(
            zip(pdf_paths, annotation_paths),
            total=len(pdf_paths),
            desc=f"{self.name}: Creating dataset",
            leave=False,
        ):
            doc_annotation = load_json(annotation_path)
            metadata = doc_annotation["metadata"]
            page_sizes = metadata["page_sizes_at_200dpi"]
            field_extractions = doc_annotation["field_extractions"]
            fields_data_by_label = {}
            for field_extraction in field_extractions:
                bbox = field_extraction[
                    "bbox"
                ]  # left, top, right, bottom in relative coordinates
                bbox = [
                    bbox[0] * page_sizes[field_extraction["page"]][0],
                    bbox[1] * page_sizes[field_extraction["page"]][1],
                    bbox[2] * page_sizes[field_extraction["page"]][0],
                    bbox[3] * page_sizes[field_extraction["page"]][1],
                ]
                bbox = BBox(
                    x1=int(bbox[0]),
                    y1=int(bbox[1]),
                    x2=int(bbox[2]),
                    y2=int(bbox[3]),
                )
                field_data = Field(
                    label=field_extraction["fieldtype"],
                    value=field_extraction["text"],
                    page_number=field_extraction["page"],
                    bbox=bbox,
                )
                if field_data.label not in fields_data_by_label:
                    fields_data_by_label[field_data.label] = field_data
                else:
                    if isinstance(fields_data_by_label[field_data.label].value, list):
                        fields_data_by_label[field_data.label].value.append(  # type: ignore[union-attr]
                            field_data.value  # type: ignore[arg-type]
                        )
                    else:
                        fields_data_by_label[field_data.label].value = [
                            fields_data_by_label[field_data.label].value,  # type: ignore[list-item]
                            field_data.value,  # type: ignore[list-item]
                        ]
            data.append(
                BenchmarkData(
                    extraction_type=ExtractionType.FIELD,
                    image_paths=pdf2image_mapping[pdf_path],
                    fields=[val for _, val in fields_data_by_label.items()],
                ),
            )
        return data
