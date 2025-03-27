from __future__ import annotations

import signal

import gradio as gr
import json_repair
import pandas as pd

from docext.core.client import sync_request
from docext.core.prompts import get_fields_confidence_score_messages
from docext.core.prompts import get_fields_messages
from docext.core.vllm import VLLMServer

fields = []
predefined_fields_for_doc_types = {
    "select_doc_type": [],
    "invoice": [
        {"field_name": "invoice_number", "description": "Invoice number"},
        {"field_name": "invoice_date", "description": "Invoice date"},
        {"field_name": "invoice_amount", "description": "Invoice amount"},
    ],
    "passport": [
        {"field_name": "full_name", "description": "Full name"},
        {"field_name": "date_of_birth", "description": "Date of birth"},
        {"field_name": "passport_number", "description": "Passport number"},
        {"field_name": "passport_type", "description": "Passport type"},
    ],
}


def add_field(field_name, description):
    global fields
    fields.append({"field_name": field_name, "description": description})
    return update_fields_display()


def update_fields_display():
    display_text = ""
    for i, field in enumerate(fields):
        display_text += f"{i}. {field['field_name']} - {field['description']}\n"
    return display_text


def clear_fields():
    global fields
    fields = []
    return update_fields_display()


def remove_field(index):
    global fields
    if 0 <= index < len(fields):
        del fields[index]
    return update_fields_display()


def add_predefined_fields(doc_type):
    global fields
    fields = predefined_fields_for_doc_types[doc_type]
    return update_fields_display()


def define_fields():
    gr.Markdown(
        """### Add all the fields you want to extract information from the documents
        - Add a field by clicking the "Add Field" button. Description is optional.
        - You can also select predefined fields for a specific document type by clicking the "Add Predefined Fields" button.
        - List of fields will be displayed below in the **`Fields`** section.
        - Remove a field by clicking the "Remove Field" button. You will need to provide the index of the field to remove.
        - Clear all the fields by clicking the "Clear All Fields" button.
        """,
    )

    with gr.Row():
        field_name = gr.Textbox(label="Field Name", placeholder="Enter field name")
        description = gr.Textbox(label="Description", placeholder="Enter description")

    with gr.Row():
        add_btn = gr.Button("Add Field")
        clear_btn = gr.Button("Clear All Fields")
        add_predefined_fields_btn = gr.Dropdown(
            choices=list(predefined_fields_for_doc_types.keys()),
            label="Select Document Type",
        )

    fields_display = gr.Textbox(label="Fields", interactive=False, lines=8)

    with gr.Row():
        field_index = gr.Number(label="Field Index to Remove", value=0, precision=0)
        remove_btn = gr.Button("Remove Field")

    add_btn.click(add_field, [field_name, description], fields_display)
    clear_btn.click(clear_fields, None, fields_display)
    remove_btn.click(remove_field, field_index, fields_display)
    add_predefined_fields_btn.select(
        add_predefined_fields,
        add_predefined_fields_btn,
        fields_display,
    )


def extract_information(file_inputs: list[str], model_name: str):
    file_paths = [file_input[0] for file_input in file_inputs]
    global fields
    field_names = [field["field_name"] for field in fields]
    fields_description = [field["description"] for field in fields]
    messages = get_fields_messages(field_names, fields_description, file_paths)
    print("sending request to vllm")
    response = sync_request(messages, model_name)["choices"][0]["message"]["content"]
    print(response)
    messages = get_fields_confidence_score_messages(messages, response, field_names)
    response_conf_score = sync_request(messages, model_name)["choices"][0]["message"][
        "content"
    ]
    print(response_conf_score)
    extracted_fields = json_repair.loads(response)
    conf_scores = json_repair.loads(response_conf_score)

    df = pd.DataFrame(
        {
            "fields": field_names,
            "answer": [extracted_fields.get(field, "") for field in field_names],
            "confidence_score": [conf_scores.get(field, 0) for field in field_names],
        },
    )
    return df


def gradio_app(model_name):
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("Information Extraction from documents"):
                instructions_md = """## Instructions
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
                gr.Markdown("""### Upload images and extract information""")
                with gr.Row():
                    with gr.Column():
                        # Create a hidden textbox for model_name
                        model_input = gr.Textbox(value=model_name, visible=False)
                        interface = gr.Interface(
                            fn=extract_information,
                            inputs=[
                                gr.Gallery(label="Upload images", preview=True),
                                model_input,
                            ],
                            outputs=gr.Dataframe(label="Extracted Information"),
                            flagging_mode="never",
                        )

        demo.launch(
            auth=("admin", "admin"),
            share=True,
            server_name="0.0.0.0",
            server_port=7861,
        )


if __name__ == "__main__":

    # get the model name from the user
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    ## start the vllm server
    vllm_server = VLLMServer(model_name)
    vllm_server.run_in_background()

    # Stop the server when the script exits
    def cleanup(signum, frame):
        print("\nReceived exit signal. Stopping vLLM server...")
        vllm_server.stop_server()
        exit(0)

    # Handle termination signals to stop the server gracefully
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        gradio_app(model_name)
    except KeyboardInterrupt:
        cleanup(None, None)
        pass
