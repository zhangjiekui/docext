from __future__ import annotations

import json
import os
from typing import List

from pdf2image import convert_from_path


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def convert_pdf2image(pdf_path: str, output_dir: str):
    """
    Convert a pdf file to a list of image files.
    Args:
        pdf_path: str, path to the pdf file. eg: "document.pdf"
        output_dir: str, path to the output directory. eg: "document_images"
    Returns:
        save_paths: list, paths to the image files. eg: ["document_images/document_0.jpeg", "document_images/document_1.jpeg", ...]
    """
    images = convert_from_path(pdf_path)
    base_filename = os.path.basename(pdf_path)
    save_paths = []
    for i, image in enumerate(images):
        save_path = os.path.join(output_dir, f"{base_filename}_{i}.jpeg")
        image.save(save_path, "JPEG")
        save_paths.append(save_path)
    return save_paths


def polygon_to_bbox(box: list[int]):
    """
    Convert 8-point polygon [x1, y1, x2, y2, x3, y3, x4, y4]
    to axis-aligned bounding box [x_min, y_min, x_max, y_max]
    Args:
        box: list of 8 integers.
    Returns:
        bbox: list of 4 integers.
    """
    x_coords = box[0::2]  # [x1, x2, x3, x4]
    y_coords = box[1::2]  # [y1, y2, y3, y4]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return [x_min, y_min, x_max, y_max]


def get_enclosing_bbox(bboxes: list[list[int]]):
    """
    Get the enclosing bounding box of a list of bounding boxes.
    Args:
        bboxes: list of bounding boxes.
    Returns:
        enclosing_bbox: list of bounding box.
    """
    if len(bboxes) == 0:
        return []
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[2] for b in bboxes)
    y_max = max(b[3] for b in bboxes)
    return [x_min, y_min, x_max, y_max]
