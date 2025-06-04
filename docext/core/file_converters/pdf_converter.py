from __future__ import annotations

from docext.core.file_converters.file_converter import FileConverter
from pdf2image import convert_from_path


class PDFConverter(FileConverter):
    def convert_to_images(self, file_path: str):
        return convert_from_path(file_path)