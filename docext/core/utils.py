from __future__ import annotations

import base64
import io
import os
from typing import Union

import pandas as pd
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def validate_fields_and_tables(fields_and_tables: dict | pd.DataFrame):

    # if its a dataframe, convert it to a dict
    if isinstance(fields_and_tables, pd.DataFrame):
        if "index" in fields_and_tables.columns:
            fields_and_tables.drop(columns=["index"], inplace=True)
        fields_df = fields_and_tables[fields_and_tables.type == "field"]
        tables_df = fields_and_tables[fields_and_tables.type == "table"]

        fields_and_tables = {
            "fields": fields_df.to_dict(orient="records"),
            "tables": tables_df.to_dict(orient="records"),
        }

    assert (
        "fields" in fields_and_tables
    ), "Fields must be present. Give empty list if no fields are present."
    assert (
        "tables" in fields_and_tables
    ), "Tables must be present. Give empty list if no tables are present."

    assert all(
        ["name" in field_details for field_details in fields_and_tables["fields"]],
    ), "All fields must have a name"
    assert all(
        ["name" in table_details for table_details in fields_and_tables["tables"]],
    ), "All tables must have a name"

    return fields_and_tables


def resize_images(file_paths: list[str], max_img_size: int):
    for file_path in file_paths:
        img = Image.open(file_path)
        img = img.resize((max_img_size, max_img_size))
        img.save(file_path)


def validate_file_paths(file_paths: list[str]):
    # TODO: add support for s3 image urls
    for file_path in file_paths:
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        assert os.path.isfile(file_path), f"File {file_path} is not a file"
        assert os.path.splitext(file_path)[1].lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".bmp",
            ".gif",
        ], f"File {file_path} is not an image"
