# Intelligent Document Processing Leaderboard

_A unified benchmark for OCR, KIE, classification, visual QA, table extraction, and confidence evaluation._

🔗 [Datasets](https://huggingface.co/collections/nanonets/idp-leaderboard-681b6fe400a6c4d8976164bc) &nbsp;&nbsp;&nbsp; 📄 [Paper](#) &nbsp;&nbsp;&nbsp; 📝 [Benchmark Details](https://idp-leaderboard.org/details/) &nbsp;&nbsp;&nbsp; 💻 [Codebase](https://github.com/NanoNets/docext) &nbsp;&nbsp;&nbsp; 🏆 [Leaderboard](https://idp-leaderboard.org)

---

## 🧠 Overview

The **Intelligent Document Processing (IDP) Leaderboard** is a standardized evaluation suite for Vision-Language Models (VLMs) focused on document understanding. It enables consistent benchmarking across a range of real-world document AI tasks.

---

## 📌 Evaluation Tasks

This benchmark evaluates performance across seven key document intelligence challenges:

- **Key Information Extraction (KIE)**: Extract structured fields from unstructured document text.
- **Visual Question Answering (VQA)**: Assess understanding of document content via question-answering.
- **Optical Character Recognition (OCR)**: Measure accuracy in recognizing printed and handwritten text.
- **Document Classification**: Evaluate how accurately models categorize various document types.
- **Long Document Processing**: Test models' reasoning over lengthy, context-rich documents.
- **Table Extraction**: Benchmark structured data extraction from complex tabular formats.
- **Confidence Score Calibration**: Evaluate the reliability and confidence of model predictions.

🔍 For in-depth information, see the [release blog](https://github.com/NanoNets/docext/tree/main/docext/benchmark).

📊 **Live leaderboard:** [https://idp-leaderboard.org](https://idp-leaderboard.org)

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/NanoNets/docext.git
cd docext
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Configure the Benchmark

Edit the configuration file at `configs/benchmark.yaml` to specify:
- **`tasks`**: Select which tasks to evaluate.
- **`datasets`**: Specify the datasets to use. By default, all datasets relevant to the selected tasks will be included.
- **`models`**: List the models you want to benchmark.
- **`max_samples_per_dataset`**: Limit the number of samples used per dataset for faster testing.
- **`max_workers`**: Set the maximum number of concurrent requests sent to the model.
- **Task-specific settings**: Adjust additional parameters depending on the task requirements.

### 2. Run the Benchmark

```bash
python docext/benchmark/benchmark.py
```

### 3. View Results

- Accuracy and cost metrics are saved as `accuracy.csv` and `cost.csv` in the working directory.
- Cached model outputs are stored in the directory set by `cache_dir` in the config. Default cache dir is `docext_benchmark_cache` (You can change from config.).

---

## 📂 Supported Tasks & Datasets

- **KIE**: DocILE, Nanonets KIE, Handwritten Forms
- **OCR**: Handwriting, Rotated Text, Digital OCR Diacritics
- **VQA**: ChartQA, DocVQA
- **LongDocBench**: NanonetsLongDocBench
- **Classification**: Nanonets Classification
- **Table Extraction**: Structured and unstructured table datasets

🔧 See [`docext/benchmark/tasks.py`](https://github.com/NanoNets/docext/blob/main/docext/benchmark/tasks.py) and `configs/benchmark.yaml` for the complete list.

---

## 📚 Citation
```
@misc{IDPLeaderboard,
  title={IDPLeaderboard: A Unified Leaderboard for Intelligent Document Processing Tasks},
  author={Souvik Mandal and Nayancy Gupta and Ashish Talewar and Paras Ahuja and Prathamesh Juvatkar and Gourinath Banda},
  howpublished={https://idp-leaderboard.org},
  year={2025},
}
```
