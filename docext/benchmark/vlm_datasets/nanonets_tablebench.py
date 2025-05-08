from __future__ import annotations

import os
from io import StringIO

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from docext.benchmark.vlm_datasets.ds import BenchmarkData
from docext.benchmark.vlm_datasets.ds import BenchmarkDataset
from docext.benchmark.vlm_datasets.ds import ExtractionType
from docext.benchmark.vlm_datasets.ds import Table


class NanonetsTableBench(BenchmarkDataset):
    name = "nanonets_tablebench"
    task = "TABLE"

    def __init__(
        self,
        hf_name: str = "Souvik3333/table_test",
        test_split: str = "test",
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        cache_dir = self._get_cache_dir(self.name, cache_dir)
        data: list[BenchmarkData] = self._load_data(
            hf_name, test_split, max_samples, cache_dir
        )
        super().__init__(self.name, data, cache_dir)

    def parse_annotations(self, annotations: str) -> pd.DataFrame:
        return pd.read_json(StringIO(annotations), orient="records")

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
                self.bytes_to_image(data_point["images"]),
                self.parse_annotations(data_point["annotation"]),
            )
            # save the image
            image_path = os.path.join(cache_dir, f"{i}.png")
            image.save(image_path)

            data.append(
                BenchmarkData(
                    image_paths=[image_path],
                    extraction_type=ExtractionType.TABLE,
                    tables=[
                        Table(
                            table=label,
                            columns=label.columns.tolist(),
                        )
                    ],
                ),
            )
        return data


class NanonetsTableBenchSmallDenseStructuredTable(NanonetsTableBench):
    name = "nanonets_small_dense_structured_table"


class NanonetsTableBenchSmallSparseStructuredTable(NanonetsTableBench):
    name = "nanonets_small_sparse_structured_table"


class NanonetsTableBenchSmallSparseUntructuredTable(NanonetsTableBench):
    name = "nanonets_small_sparse_unstructured_table"


class NanonetsTableBenchLongDenseStructuredTable(NanonetsTableBench):
    name = "nanonets_long_dense_structured_table"


class NanonetsTableBenchLongSparseStructuredTable(NanonetsTableBench):
    name = "nanonets_long_sparse_structured_table"


class NanonetsTableBenchLongSparseUntructuredTable(NanonetsTableBench):
    name = "nanonets_long_sparse_unstructured_table"
