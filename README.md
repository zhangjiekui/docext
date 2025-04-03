# docext

An on-premises document information extraction tool powered by vision-language models.

![Demo](assets/demo.png)

## Overview

docext is a powerful tool for extracting structured information from documents such as invoices, passports, and other forms. It leverages vision-language models (VLMs) to accurately identify and extract both field data and tabular information from document images.


## Features

- **User-friendly interface**: Built with Gradio for easy document processing
- **Flexible extraction**: Define custom fields or use pre-built templates
- **Table extraction**: Extract structured tabular data from documents
- **Confidence scoring**: Get confidence levels for extracted information
- **On-premises deployment**: Run entirely on your own infrastructure
- **Multi-page support**: Process documents with multiple pages
- **REST API**: Programmatic access for integration with your applications
- **Pre-built templates**: Ready-to-use templates for common document types:
  - Invoices
  - Passports
  - Create your own templates

## Quickstart
- [Colab notebook](https://colab.research.google.com/drive/1r1asxGeezfWnJvw8jimfFAB2sGjk1HdM?usp=sharing)
- [Docker](#Docker)

## Installation

```bash
# create a virtual environment
## install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh
## create a virtual environment with python 3.11
uv venv --python=3.11
source .venv/bin/activate

# Install from PyPI
uv pip install docext

# Or install from source
git clone https://github.com/nanonets/docext.git
cd docext
uv pip install -e .
```

## Web Interface

docext includes a Gradio-based web interface for easy document processing:

```bash
# Start the web interface with default configs
python -m docext.app.app

# Start the web interface with custom configs
python -m docext.app.app --model_name "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" --max_img_size 1024 # `--help` for more options
```

The interface will be available at `http://localhost:7860` with default credentials: (You can change the port by using `--ui_port` flag)

- Username: `admin`
- Password: `admin`

## API access

docext also provides a REST API for programmatic access to the document extraction functionality.
1. start the API server
```bash
# increase the concurrency limit to process more requests in parallel, default is 1
python -m docext.app.app --concurrency_limit 10
```

2. use the API to extract information from a document
```python
import pandas as pd
import concurrent.futures
from gradio_client import Client, handle_file


def dataframe_to_custom_dict(df: pd.DataFrame) -> dict:
    return {
        "headers": df.columns.tolist(),
        "data": df.values.tolist(),
        "metadata": None  # Modify if metadata is needed
    }

def dict_to_dataframe(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d["data"], columns=d["headers"])


def get_extracted_fields_and_tables(
    client_url: str,
    username: str,
    password: str,
    model_name: str,
    fields_and_tables: dict,
    file_inputs: list[dict]
):
    client = Client(client_url, auth=(username, password))
    result = client.predict(
        file_inputs=file_inputs,
        model_name=model_name,
        fields_and_tables=fields_and_tables,
        api_name="/extract_information"
    )
    fields_results, tables_results = result
    fields_df = dict_to_dataframe(fields_results)
    tables_df = dict_to_dataframe(tables_results)
    return fields_df, tables_df


fields_and_tables = dataframe_to_custom_dict(pd.DataFrame([
    {"name": "invoice_number", "type": "field", "description": "Invoice number"},
    {"name": "item_description", "type": "table", "description": "Item/Product description"}
    # add more fields and table columns as needed
]))

file_inputs = [
    {
        # "image": handle_file("https://your_image_url/invoice.jpg") # incase the image is hosted on the internet
        "image": handle_file("assets/invoice_test.jpeg") # incase the image is hosted on the local machine
    }
]

## send single request
### client url can be the local host or the public url like `https://6986bdd23daef6f7eb.gradio.live/`
fields_df, tables_df = get_extracted_fields_and_tables(
    "http://localhost:7860", "admin", "admin", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", fields_and_tables, file_inputs
)
print("========Fields:=========")
print(fields_df)
print("========Tables:=========")
print(tables_df)


## send multiple requests in parallel
# Define a wrapper function for parallel execution
def run_request():
    return get_extracted_fields_and_tables(
        "http://localhost:7860", "admin", "admin", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", fields_and_tables, file_inputs
    )

# Use ThreadPoolExecutor to send 10 requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    future_results = [executor.submit(run_request) for _ in range(10)]

    for future in concurrent.futures.as_completed(future_results):
        fields_df, tables_df = future.result()
        print("========Fields:=========")
        print(fields_df)
        print("========Tables:=========")
        print(tables_df)
```

## Requirements

- Python 3.11+
- CUDA-compatible GPU (for optimal performance). Use Google Colab for free GPU access.
- Dependencies listed in requirements.txt

## Models

docext uses vision-language models for document understanding. By default, it uses: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`

Recommended models based on GPU memory:
| Model | GPU Memory |
|-------|------------|
| Qwen/Qwen2.5-VL-7B-Instruct-AWQ | 16GB |
| Qwen/Qwen2.5-VL-7B-Instruct | 24GB |
| Qwen/Qwen2.5-VL-32B-Instruct-AWQ | 48GB |
| Qwen/Qwen2.5-VL-32B-Instruct | 80 GB |

## Docker
1. Add your [huggingface token](https://huggingface.co/docs/hub/en/security-tokens) to the environment variable. Not needed if you are using the default model.
2. Utilize all available GPUs or specify a particular one as needed (e.g., --gpus '"device=0"'). CPU mode is not supported; for trying out the app, we recommend using Google Colab, which offers free GPU access.
```bash
docker run --rm \
  --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 7860:7860 \
  --ipc=host \
  --gpus all \
  docext:v0.1.2
```

## About

docext is developed by [Nanonets](https://nanonets.com), a leader in document AI and intelligent document processing solutions. Nanonets is committed to advancing the field of document understanding through open-source contributions and innovative AI technologies. If you are looking for information extraction solutions for your business, please visit [our website](https://nanonets.com) to learn more.

## Contributing

We welcome contributions! Please see [contribution.md](contribution.md) for guidelines.
If you have a feature request or need support for a new model, feel free to open an issue—we’d love to discuss it further!

## Troubleshooting

If you encounter any issues while using `docext`, please refer to our [Troubleshooting guide](Troubleshooting.md) for common problems and solutions.


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
