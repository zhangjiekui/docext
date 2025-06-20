<h1 align="center">docext</h1>
<p DemoURL：https://huggingface.co/spaces/Souvik3333/Nanonets-ocr-s>

<p align="center"><em>An on-premises document information extraction and benchmarking toolkit.</em></p>

<p align="center">
  <a href="https://pepy.tech/projects/docext">
    <img src="https://static.pepy.tech/badge/docext" alt="PyPI Downloads" />
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" />
  </a>
  <a href="https://colab.research.google.com/drive/1r1asxGeezfWnJvw8jimfFAB2sGjk1HdM?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
  <a href="https://pypi.org/project/docext/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/docext">
  </a>
</p>

<!-- ![Demo Docext](https://raw.githubusercontent.com/NanoNets/docext/main/assets/pdf2markdown.jpg) -->
![Demo Docext](assets/pdf2markdown.png)


## New Model Release: Nanonets-OCR-s

**We're excited to announce the release of Nanonets-OCR-s, a compact 3B parameter model specifically trained for efficient image to markdown conversion with semantic understanding for images, signatures, watermarks, etc.!**

  📢 [Read the full announcement](https://nanonets.com/research/nanonets-ocr-s) | 🤗 [Hugging Face model](https://huggingface.co/nanonets/Nanonets-OCR-s)

## Overview

docext is a comprehensive on-premises document intelligence toolkit powered by vision-language models (VLMs). It provides three core capabilities:

**📄 PDF & Image to Markdown Conversion**: Transform documents into structured markdown with intelligent content recognition, including LaTeX equations, signatures, watermarks, tables, and semantic tagging.

**🔍 Document Information Extraction**: OCR-free extraction of structured information (fields, tables, etc.) from documents such as invoices, passports, and other document types, with confidence scoring.

**📊 Intelligent Document Processing Leaderboard**: A comprehensive benchmarking platform that tracks and evaluates vision-language model performance across OCR, Key Information Extraction (KIE), document classification, table extraction, and other intelligent document processing tasks.


## Features
### PDF and Image to Markdown
Convert both PDF and images to markdown with content recognition and semantic tagging.
- **LaTeX Equation Recognition**: Convert both inline and block LaTeX equations in images to markdown.
- **Intelligent Image Description**: Generate a detailed description for all images in the document within `<img></img>` tags.
- **Signature Detection**: Detect and mark signatures and watermarks in the document. Signatures text are extracted within `<signature></signature>` tags.
- **Watermark Detection**: Detect and mark watermarks in the document. Watermarks text are extracted within `<watermark></watermark>` tags.
- **Page Number Detection**: Detect and mark page numbers in the document. Page numbers are extracted within `<page_number></page_number>` tags.
- **Checkboxes and Radio Buttons**: Converts form checkboxes and radio buttons into standardized Unicode symbols (☐, ☑, ☒).
- **Table Detection**: Convert complex tables into html tables.

🔍 For in-depth information, see the [release blog](https://nanonets.com/research/nanonets-ocr-s/).

For setup instructions and additional details, check out the full feature guide for the [pdf to markdown](https://github.com/NanoNets/docext/blob/main/PDF2MD_README.md).

### Intelligent Document Processing Leaderboard
This benchmark evaluates performance across seven key document intelligence challenges:

- **Key Information Extraction (KIE)**: Extract structured fields from unstructured document text.
- **Visual Question Answering (VQA)**: Assess understanding of document content via question-answering.
- **Optical Character Recognition (OCR)**: Measure accuracy in recognizing printed and handwritten text.
- **Document Classification**: Evaluate how accurately models categorize various document types.
- **Long Document Processing**: Test models' reasoning over lengthy, context-rich documents.
- **Table Extraction**: Benchmark structured data extraction from complex tabular formats.
- **Confidence Score Calibration**: Evaluate the reliability and confidence of model predictions.

🔍 For in-depth information, see the [release blog](https://idp-leaderboard.org/details/).

📊 **Live leaderboard:** [https://idp-leaderboard.org](https://idp-leaderboard.org)

For setup instructions and additional details, check out the full feature guide for the [Intelligent Document Processing Leaderboard](https://github.com/NanoNets/docext/tree/main/docext/benchmark).

### Docext
- **Flexible extraction**: Define custom fields or use pre-built templates
- **Table extraction**: Extract structured tabular data from documents
- **Confidence scoring**: Get confidence levels for extracted information
- **On-premises deployment**: Run entirely on your own infrastructure (Linux, MacOS)
- **Multi-page support**: Process documents with multiple pages
- **REST API**: Programmatic access for integration with your applications
- **Pre-built templates**: Ready-to-use templates for common document types:
  - Invoices
  - Passports
  - Add/delete new fields/columns for other templates.

For more details (Installation, Usage, and so on), please check out the [feature guide](https://github.com/NanoNets/docext/blob/main/EXT_README.md).

## Change Log

### Latest Updates
- **12-06-2025** - Added pdf and image to markdown support.
- **06-06-2025** - Added `gemini-2.5-pro-preview-06-05` evaluation metrics to the leaderboard.
- **04-06-2025** - Added support for PDF and multiple documents in `docext` extraction.

<details>
<summary>Older Changes</summary>

- **23-05-2025** – Added `gemini-2.5-pro-preview-03-25`, `claude-sonnet-4` evaluation metrics to the leaderboard.
- **17-05-2025** – Added `InternVL3-38B-Instruct`, `qwen2.5-vl-32b-instruct` evaluation metrics to the leaderboard.
- **16-05-2025** – Added `gemma-3-27b-it` evaluation metrics to the leaderboard.
- **12-05-2025** – Added `Claude 3.7 sonnet`, `mistral-medium-3` evaluation metrics to the leaderboard.
</details>

## About

docext is developed by [Nanonets](https://nanonets.com), a leader in document AI and intelligent document processing solutions. Nanonets is committed to advancing the field of document understanding through open-source contributions and innovative AI technologies. If you are looking for information extraction solutions for your business, please visit [our website](https://nanonets.com) to learn more.

## Contributing

We welcome contributions! Please see [contribution.md](https://github.com/NanoNets/docext/blob/main/contribution.md) for guidelines.
If you have a feature request or need support for a new model, feel free to open an issue—we'd love to discuss it further!

## Troubleshooting

If you encounter any issues while using `docext`, please refer to our [Troubleshooting guide](https://github.com/NanoNets/docext/blob/main/Troubleshooting.md) for common problems and solutions.


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
