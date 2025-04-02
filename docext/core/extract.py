from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from typing import Union

import json_repair
import mdpd
import pandas as pd
from loguru import logger

from docext.core.client import sync_request
from docext.core.prompts import get_fields_confidence_score_messages
from docext.core.prompts import get_fields_messages
from docext.core.prompts import get_tables_messages
from docext.core.utils import resize_images
from docext.core.utils import validate_fields_and_tables
from docext.core.utils import validate_file_paths


def extract_fields_from_documents(
    file_paths: list[str],
    model_name: str,
    fields: list[dict],
):
    if len(fields) == 0:
        return pd.DataFrame()
    field_names = [field["name"] for field in fields]
    fields_description = [field.get("description", "") for field in fields]
    messages = get_fields_messages(field_names, fields_description, file_paths)

    logger.info(f"Sending request to {model_name}")
    response = sync_request(messages, model_name)["choices"][0]["message"]["content"]
    logger.info(f"Response: {response}")

    # conf score
    messages = get_fields_confidence_score_messages(messages, response, field_names)
    response_conf_score = sync_request(messages, model_name)["choices"][0]["message"][
        "content"
    ]
    logger.info(f"Response conf score: {response_conf_score}")

    extracted_fields = json_repair.loads(response)
    conf_scores = json_repair.loads(response_conf_score)

    df = pd.DataFrame(
        {
            "fields": field_names,
            "answer": [extracted_fields.get(field, "") for field in field_names],
            "confidence": [conf_scores.get(field, 0) for field in field_names],
        },
    )
    return df


def extract_tables_from_documents(
    file_paths: list[str],
    model_name: str,
    columns: list[dict],
):
    if len(columns) == 0:
        return pd.DataFrame()
    columns_names = [column["name"] for column in columns if column["type"] == "table"]
    columns_description = [
        column.get("description", "") for column in columns if column["type"] == "table"
    ]
    messages = get_tables_messages(columns_names, columns_description, file_paths)

    logger.info(f"Sending request to {model_name}")
    response = sync_request(messages, model_name)["choices"][0]["message"]["content"]
    logger.info(f"Response: {response}")

    df = mdpd.from_md(response)

    return df


def extract_information(
    file_inputs: list[tuple],
    model_name: str,
    max_img_size: int,
    fields_and_tables: dict[str, list[dict]] | pd.DataFrame,
):
    fields_and_tables = validate_fields_and_tables(fields_and_tables)
    if len(fields_and_tables["fields"]) == 0 and len(fields_and_tables["tables"]) == 0:
        return pd.DataFrame(), pd.DataFrame()
    file_paths: list[str] = [
        file_input[0] if isinstance(file_input, tuple) else file_input
        for file_input in file_inputs
    ]
    validate_file_paths(file_paths)
    resize_images(file_paths, max_img_size)

    # call fields and tables extraction in parallel
    with ThreadPoolExecutor() as executor:
        future_fields = executor.submit(
            extract_fields_from_documents,
            file_paths,
            model_name,
            fields_and_tables["fields"],
        )
        future_tables = executor.submit(
            extract_tables_from_documents,
            file_paths,
            model_name,
            fields_and_tables["tables"],
        )

        fields_df = future_fields.result()
        tables_df = future_tables.result()
    return fields_df, tables_df
