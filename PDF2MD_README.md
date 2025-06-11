# PDF to Markdown API Documentation

Convert PDF documents and images to high-quality markdown format using vision-language models with real-time streaming support.

## Features

- **Multi-format Support**: PDF, JPG, JPEG, PNG, TIFF, BMP, GIF, WebP
- **Real-time Streaming**: Progressive conversion with live updates
- **Page-by-page Processing**: Efficient memory usage and granular progress
- **Multi-page Documents**: Automatic page separation with horizontal rules
- **Mathematical Equations**: LaTeX notation support
- **Table Conversion**: Automatic markdown table formatting
- **Structure Preservation**: Headers, formatting, and logical reading order

## Quick Start

### 1. Start the API Server

```bash
# Start with default settings
python -m docext.app.app

# Start with custom configuration
python -m docext.app.app --model_name hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ --max_img_size 2048 --concurrency_limit 5
```

### 2. Access the Web Interface

Navigate to `http://localhost:7860` and use the **"Image and PDF to markdown"** tab.

- Username: `admin`
- Password: `admin`

## API Access

```python
import time
from gradio_client import Client, handle_file

def convert_pdf_to_markdown(
    client_url: str,
    username: str,
    password: str,
    file_paths: list[str],
    model_name: str = "hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
):
    """
    Convert PDF/images to markdown using the API

    Args:
        client_url: URL of the docext server
        username: Authentication username
        password: Authentication password
        file_paths: List of file paths to convert
        model_name: Model to use for conversion

    Returns:
        str: Converted markdown content
    """
    client = Client(client_url, auth=(username, password))

    # Prepare file inputs
    file_inputs = [{"image": handle_file(file_path)} for file_path in file_paths]

    # Convert to markdown (non-streaming)
    result = client.predict(
        images=file_inputs,
        api_name="/process_markdown_streaming"
    )

    return result

# Example usage
CLIENT_URL = "http://localhost:7860"

# Single image conversion
markdown_content = convert_pdf_to_markdown(
    CLIENT_URL,
    "admin",
    "admin",
    ["assets/invoice_test.pdf"]
)
print(markdown_content)

# Multiple files conversion
markdown_content = convert_pdf_to_markdown(
    CLIENT_URL,
    "admin",
    "admin",
    ["assets/invoice_test.jpeg", "assets/invoice_test.pdf"]
)
print(markdown_content)
```
