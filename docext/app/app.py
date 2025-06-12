from __future__ import annotations

import os
import signal

import gradio as gr
import pandas as pd
from loguru import logger

from docext.app.args import parse_args
from docext.app.pdf2md import pdf_to_markdown_ui
from docext.app.utils import check_ollama_healthcheck
from docext.app.utils import check_vllm_healthcheck
from docext.app.utils import cleanup
from docext.core.config import TEMPLATES_FIELDS
from docext.core.config import TEMPLATES_TABLES
from docext.core.extract import extract_information
from docext.core.utils import convert_files_to_images
from docext.core.vllm import VLLMServer

METADATA = []


def add_field(field_name: str, type: str, description: str):
    global METADATA
    METADATA.append(
        {"field_name": field_name, "type": type, "description": description},
    )
    return update_fields_display()


def update_fields_display():
    dict_data = {"index": [], "type": [], "name": [], "description": []}
    for i, metadata in enumerate(METADATA):
        dict_data["index"].append(i)
        dict_data["type"].append(metadata["type"])
        dict_data["name"].append(metadata["field_name"])
        dict_data["description"].append(metadata["description"])
    return pd.DataFrame(dict_data)


def clear_fields():
    global METADATA
    METADATA = []
    return update_fields_display()


def remove_field(index):
    global METADATA
    if 0 <= index < len(METADATA):
        del METADATA[index]
    return update_fields_display()


def add_predefined_fields(doc_type):
    global METADATA
    fields = TEMPLATES_FIELDS.get(doc_type, [])
    fields = [
        {
            "field_name": field["field_name"],
            "type": "field",
            "description": field["description"],
        }
        for field in fields
    ]
    tables = TEMPLATES_TABLES.get(doc_type, [])
    tables = [
        {
            "field_name": table["field_name"],
            "type": "table",
            "description": table["description"],
        }
        for table in tables
    ]
    METADATA = fields + tables
    return update_fields_display()


def define_keys_and_extract(model_name: str, max_img_size: int, concurrency_limit: int):
    gr.Markdown(
        """### Add all the fields you want to extract information from the documents
        - Add a field by clicking the **`Add Field`** button. Description is optional.
        - You can also select predefined fields for a specific document type by selecting a template in **`Existing Templates`** dropdown.
        - List of fields will be displayed below in the **`Fields`** section.
        - Remove a field by clicking the **`Remove Field`** button. You will need to provide the index of the field to remove.
        - Clear all the fields by clicking the **`Clear All Fields`** button.
        """,
    )
    with gr.Row():
        add_predefined_fields_btn = gr.Dropdown(
            choices=["Select a template"] + list(TEMPLATES_FIELDS.keys()),
            label="Existing Templates",
        )

    gr.Markdown("""#### Add a new field/column""")
    with gr.Row():
        field_name = gr.Textbox(
            label="Field Name",
            placeholder="Enter field/column name",
        )
        type = gr.Dropdown(choices=["field", "table"], label="Type")
        description = gr.Textbox(label="Description", placeholder="Enter description")

    with gr.Row():
        add_btn = gr.Button("Add Field/Column ✚")
        clear_btn = gr.Button("Clear All Fields/Columns ❌")

    fields_display = gr.Dataframe(
        label="Fields/Columns",
        wrap=True,
        interactive=False,
        headers=["index", "type", "name", "description"],
    )

    gr.Markdown("""#### Remove a field/column""")
    with gr.Row():
        field_index = gr.Number(
            label="Field/Column Index to Remove",
            value=0,
            precision=0,
        )
        remove_btn = gr.Button("Remove Field/Column −")

    add_btn.click(add_field, [field_name, type, description], fields_display)
    clear_btn.click(clear_fields, None, fields_display)
    remove_btn.click(remove_field, field_index, fields_display)
    add_predefined_fields_btn.select(
        add_predefined_fields,
        add_predefined_fields_btn,
        fields_display,
    )

    gr.Markdown("""-----------------------------------------""")
    gr.Markdown("""### Upload images and extract information ⚙️""")
    with gr.Row():
        with gr.Column():
            # Create a hidden textbox for model_name
            model_input = gr.Textbox(value=model_name, visible=False)
            max_img_size_input = gr.Number(
                value=max_img_size,
                visible=False,
            )

            file_input = gr.File(
                label="Upload Documents",
                file_types=[
                    ".pdf",
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tiff",
                    ".bmp",
                    ".gif",
                    ".webp",
                ],
                file_count="multiple",
            )
            images_input = gr.Gallery(
                label="Document Preview", preview=True, visible=False
            )
            submit_btn = gr.Button("Submit", visible=False)

            def handle_file_upload(files):
                if not files:
                    return None, gr.update(visible=False), gr.update(visible=False)

                file_paths = [f.name for f in files]
                # Convert PDFs to images if necessary and get all image paths
                image_paths = convert_files_to_images(file_paths)
                return (
                    image_paths,
                    gr.update(visible=True, value=image_paths),
                    gr.update(visible=True),
                )

            file_input.change(
                handle_file_upload,
                inputs=[file_input],
                outputs=[images_input, images_input, submit_btn],
            )

    with gr.Row():
        with gr.Column(scale=3):
            extracted_fields_output = gr.Dataframe(
                label="Extracted Fields",
                wrap=True,
                interactive=False,
                headers=["fields", "answer", "confidence"],
            )
        with gr.Column(scale=7):
            extracted_tables_output = gr.Dataframe(
                label="Extracted Tables",
                wrap=True,
                interactive=False,
                headers=["col1", "col2", "coln"],
            )

    submit_btn.click(
        extract_information,
        [images_input, model_name, max_img_size, fields_display],
        [extracted_fields_output, extracted_tables_output],
        concurrency_limit=concurrency_limit,
    )


def gradio_app(
    model_name: str,
    gradio_port: int,
    max_img_size: int,
    concurrency_limit: int,
    share: bool,
    vllm_server_host: str,
    vllm_server_port: int,
    max_gen_tokens: int,
):
    # set vlm_model_url env variable
    hosted_model_url = f"http://{vllm_server_host}:{vllm_server_port}"
    os.environ["VLM_MODEL_URL"] = (
        f"{hosted_model_url}/v1"
        if model_name.startswith("hosted_vllm/")
        else hosted_model_url
    )

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("Information Extraction from documents"):
                instructions_md = """## Instructions 📖
                - Define the fields (and optionally the description) you want to extract from the document.
                - Upload a document (can be a single image or a list of images in case of multipage document).
                - Currently, we support only image files.
                - Currently, we only support maximum 5 images at a time. You can change this limit following the advanced instructions.
                -----------------------------------------
                """
                gr.Markdown(instructions_md)
                # Define the fields
                model_input = gr.Textbox(value=model_name, visible=False)
                max_img_size_input = gr.Number(
                    value=max_img_size,
                    visible=False,
                )
                define_keys_and_extract(
                    model_input,
                    max_img_size_input,
                    concurrency_limit,
                )
            with gr.Tab("Image and PDF to markdown"):
                gr.Markdown(
                    """Upload an image or a PDF file and convert it to markdown."""
                )
                pdf_to_markdown_ui(
                    model_name, max_img_size, concurrency_limit, max_gen_tokens
                )

        logger.info(f"Launching gradio app on port {gradio_port}")
        demo.launch(
            auth=("admin", "admin"),
            share=not share,
            server_name="0.0.0.0",
            server_port=gradio_port,
            show_error=True,
        )


def main(
    model_name: str,
    host: str,
    port: int,
    gradio_port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_imgs: int,
    vllm_start_timeout: int,
    max_img_size: int,
    concurrency_limit: int,
    share: bool,
    dtype: str,
    max_gen_tokens: int,
):
    vllm_server = None
    if model_name.startswith("hosted_vllm/") and (
        "localhost" in host or host == "0.0.0.0" or host == "127.0.0.1"
    ):
        # check if the vllm server is running on the given host and port
        if check_vllm_healthcheck(host, port):
            logger.info(f"vLLM server is running on {host}:{port}")
        else:
            logger.warning(
                f"vLLM server is not running on {host}:{port}. Starting vLLM server...",
            )
            vllm_server = VLLMServer(
                model_name=model_name,
                host=host,
                port=port,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_imgs=max_num_imgs,
                vllm_start_timeout=vllm_start_timeout,
                dtype=dtype,
            )
            vllm_server.run_in_background()

            # Handle termination signals to stop the server gracefully
            signal.signal(
                signal.SIGINT,
                lambda signum, frame: cleanup(signum, frame, vllm_server),
            )
            signal.signal(
                signal.SIGTERM,
                lambda signum, frame: cleanup(signum, frame, vllm_server),
            )

        logger.info(f"Using local model. Current model: {model_name}")
    elif model_name.startswith("ollama/"):
        # check if the ollama server is running on the given host and port
        if check_ollama_healthcheck(host, port):
            logger.info(f"OLLAMA server is running on {host}:{port}")
        elif check_ollama_healthcheck("localhost", 11434) and (
            host == "localhost" or host == "127.0.0.1" or host == "0.0.0.0"
        ):
            # common mistake, people forget to change the port for ollama server
            logger.warning(
                f"OLLAMA server is running on localhost:11434. Changed the `--vlm_server_port` to 11434",
            )
            port = 11434
        else:
            logger.error(
                f"OLLAMA server is not running on {host}:{port}. Please install and start the server following the instructions in the Wiki.",
            )
            exit(1)
    else:
        logger.info(f"Not using local model. Current model: {model_name}")

    try:
        gradio_app(
            model_name,
            gradio_port,
            max_img_size,
            concurrency_limit,
            share,
            host,
            port,
            max_gen_tokens,
        )
    except (KeyboardInterrupt, Exception) as e:
        logger.error(f"Error: {e}")
        if vllm_server:
            cleanup(None, None, vllm_server)


def docext_app():
    args = parse_args()
    logger.info(f"Config:\n{args}")

    main(
        args.model_name,
        args.vlm_server_host,
        args.vlm_server_port,
        args.ui_port,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.max_num_imgs,
        args.vllm_start_timeout,
        args.max_img_size,
        args.concurrency_limit,
        args.share,
        args.dtype,
        args.max_gen_tokens,
    )


if __name__ == "__main__":
    docext_app()
