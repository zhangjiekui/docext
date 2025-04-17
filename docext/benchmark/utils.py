from __future__ import annotations

import base64

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
