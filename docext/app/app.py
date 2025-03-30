from __future__ import annotations

import signal
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from PIL import Image

from docext.app.args import parse_args
from docext.app.utils import cleanup
from docext.core.config import TEMPLATES_FIELDS
from docext.core.config import TEMPLATES_TABLES
from docext.core.extract import extract_fields_from_documents
from docext.core.extract import extract_tables_from_documents
from docext.core.vllm import VLLMServer

# from docext.core.prompts import get_fields_bboxes_messages

METADATA = []


def add_field(field_name: str, type: str, description: str):
    global METADATA
    METADATA.append(
        {"field_name": field_name, "type": type, "description": description},
    )
    return update_fields_display()


def update_fields_display():
    display_text = ""
    dict_data = {"index": [], "type": [], "name": [], "description": []}
    for i, metadata in enumerate(METADATA):
        # display_text += f"{i}. {metadata['type']} - {metadata['field_name']} - {metadata['description']}\n"
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


def define_fields():
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


def extract_information(file_inputs: list[str], model_name: str, max_img_size: int):
    file_paths: list[str] = [file_input[0] for file_input in file_inputs]
    for file_path in file_paths:
        img = Image.open(file_path)
        img = img.resize((max_img_size, max_img_size))
        img.save(file_path)
    fields: list[dict] = [field for field in METADATA if field["type"] == "field"]
    tables: list[dict] = [field for field in METADATA if field["type"] == "table"]
    # call fields and tables extraction in parallel
    fields_df: pd.DataFrame = extract_fields_from_documents(
        file_paths,
        model_name,
        fields,
    )
    tables_df: pd.DataFrame = extract_tables_from_documents(
        file_paths,
        model_name,
        tables,
    )
    # with ThreadPoolExecutor() as executor:
    #     future_fields = executor.submit(
    #         extract_fields_from_documents,
    #         file_paths,
    #         model_name,
    #         fields,
    #     )
    #     future_tables = executor.submit(
    #         extract_tables_from_documents,
    #         file_paths,
    #         model_name,
    #         tables,
    #     )

    #     fields_df = future_fields.result()
    #     tables_df = future_tables.result()
    return fields_df, tables_df


def gradio_app(model_name: str, gradio_port: int, max_img_size: int):
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
                with gr.Row():
                    with gr.Column():
                        define_fields()

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
                        interface = gr.Interface(
                            fn=extract_information,
                            inputs=[
                                gr.Gallery(label="Upload images", preview=True),
                                model_input,
                                max_img_size_input,
                            ],
                            outputs=[
                                gr.Dataframe(
                                    label="Extracted Information",
                                    wrap=True,
                                    interactive=False,
                                    column_widths=["100px", "140px", "70px"],
                                ),
                                gr.Dataframe(
                                    label="Extracted Tables",
                                    wrap=True,
                                    interactive=False,
                                ),
                            ],
                            flagging_mode="never",
                        )

        demo.launch(
            auth=("admin", "admin"),
            share=True,
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
        gradio_app(model_name, gradio_port, max_img_size)
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
    )


if __name__ == "__main__":
    docext_app()
