from __future__ import annotations

import re
import time

import gradio as gr

from docext.core.pdf2md.pdf2md import convert_to_markdown_stream
from docext.core.utils import convert_files_to_images


def process_tags(content: str) -> str:
    content = content.replace("<img>", "&lt;img&gt;")
    content = content.replace("</img>", "&lt;/img&gt;")
    content = content.replace("<watermark>", "&lt;watermark&gt;")
    content = content.replace("</watermark>", "&lt;/watermark&gt;")
    content = content.replace("<page_number>", "&lt;page_number&gt;")
    content = content.replace("</page_number>", "&lt;/page_number&gt;")
    content = content.replace("<signature>", "&lt;signature&gt;")
    content = content.replace("</signature>", "&lt;/signature&gt;")

    return content


def pdf_to_markdown_ui(model_name: str, max_img_size: int, concurrency_limit: int):
    with gr.Row():
        with gr.Column():
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

            formatted_output = gr.Markdown(
                label="Formatted model prediction",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {
                        "left": "\\begin{align*}",
                        "right": "\\end{align*}",
                        "display": True,
                    },
                ],
                line_breaks=True,
                show_copy_button=True,
            )

            def process_markdown_streaming(images):
                """
                Process markdown with streaming updates (page by page)
                """
                # Initialize with a loading message
                num_pages = len(images) if images else 0
                if num_pages == 1:
                    yield "🔄 **Converting document to markdown...**\n\n*Processing single page document...*"
                else:
                    yield f"🔄 **Converting {num_pages}-page document to markdown...**\n\n*Processing page by page...*"

                # Stream the actual conversion
                current_page = 1
                for markdown_content in convert_to_markdown_stream(
                    images, model_name, max_img_size, concurrency_limit
                ):
                    # Add progress indicator at the top for multi-page documents
                    if num_pages > 1:
                        progress_header = f"📄 **Document Conversion Progress** (Processing page {min(current_page, num_pages)} of {num_pages})\n\n"
                        yield progress_header + process_tags(markdown_content)
                    else:
                        yield process_tags(markdown_content)

                    # Estimate current page based on content length (rough approximation)
                    if "---" in markdown_content:
                        current_page = markdown_content.count("---") + 1

                    # Small delay to make streaming visible
                    time.sleep(0.05)

            submit_btn.click(
                process_markdown_streaming,
                inputs=[images_input],
                outputs=[formatted_output],
            )
