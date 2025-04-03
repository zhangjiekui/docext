from __future__ import annotations

import signal

import gradio as gr
import pandas as pd
from loguru import logger

from docext.app.args import parse_args
from docext.app.utils import cleanup
from docext.core.config import TEMPLATES_FIELDS
from docext.core.config import TEMPLATES_TABLES
from docext.core.extract import extract_information
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

    fields_display = gr.Dataframe(label="Fields/Columns", wrap=True, interactive=False)

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

            images_input = gr.Gallery(label="Upload images", preview=True)
            submit_btn = gr.Button("Submit")

    with gr.Row():
        with gr.Column(scale=3):
            extracted_fields_output = gr.Dataframe(
                label="Extracted Fields",
                wrap=True,
                interactive=False,
            )
        with gr.Column(scale=7):
            extracted_tables_output = gr.Dataframe(
                label="Extracted Tables",
                wrap=True,
                interactive=False,
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
):
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

        demo.launch(
            auth=("admin", "admin"),
            share=not share,
            server_name="0.0.0.0",
            server_port=gradio_port,
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
):
    vllm_server = VLLMServer(
        model_name=model_name,
        host=host,
        port=port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_imgs=max_num_imgs,
        vllm_start_timeout=vllm_start_timeout,
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

    try:
        gradio_app(model_name, gradio_port, max_img_size, concurrency_limit, share)
    except KeyboardInterrupt:
        cleanup(None, None, vllm_server)
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        cleanup(None, None, vllm_server)


def docext_app():
    args = parse_args()
    logger.info(f"Config:\n{args}")

    main(
        args.model_name,
        args.vllm_host,
        args.vllm_port,
        args.ui_port,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.max_num_imgs,
        args.vllm_start_timeout,
        args.max_img_size,
        args.concurrency_limit,
        args.share,
    )


if __name__ == "__main__":
    docext_app()
