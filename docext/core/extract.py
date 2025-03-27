from __future__ import annotations

import json_repair
import mdpd
import pandas as pd
from loguru import logger

from docext.core.client import sync_request
from docext.core.prompts import get_fields_confidence_score_messages
from docext.core.prompts import get_fields_messages
from docext.core.prompts import get_tables_messages


def extract_fields_from_documents(
    file_paths: list[str],
    model_name: str,
    fields: list[dict],
):
    field_names = [field["field_name"] for field in fields]
    fields_description = [field["description"] for field in fields]
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
    columns_names = [
        column["field_name"] for column in columns if column["type"] == "table"
    ]
    columns_description = [
        column["description"] for column in columns if column["type"] == "table"
    ]
    messages = get_tables_messages(columns_names, columns_description, file_paths)

    logger.info(f"Sending request to {model_name}")
    response = sync_request(messages, model_name)["choices"][0]["message"]["content"]
    logger.info(f"Response: {response}")

    df = mdpd.from_md(response)

    return df
