from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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


def pdf_to_markdown_ui(
    model_name: str, max_img_size: int, concurrency_limit: int, max_gen_tokens: int
):
    with gr.Row():
        with gr.Column():
            # Add status indicator for concurrent processing
            gr.Markdown(
                """
Try Nanonets-OCR-s<br>
We’ve open-sourced Nanonets-OCR-s, A model for transforming documents into structured markdown with content recognition and semantic tagging.<br>
📖 [Release Blog](https://huggingface.co/nanonets/Nanonets-OCR-s) 🤗 [View on Hugging Face](https://huggingface.co/nanonets/Nanonets-OCR-s)
""",
                visible=True,
            ) if model_name != "hosted_vllm/nanonets/Nanonets-OCR-s" else None

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
                Optimized for concurrent processing of multiple requests
                """
                # Generate unique request ID for tracking
                request_id = str(uuid.uuid4())[:8]
                start_time = datetime.now().strftime("%H:%M:%S")

                # Initialize with a loading message including concurrent processing info
                num_pages = len(images) if images else 0

                # Stream the actual conversion
                current_page = 1
                try:
                    for markdown_content in convert_to_markdown_stream(
                        images,
                        model_name,
                        max_img_size,
                        concurrency_limit,
                        max_gen_tokens,
                    ):
                        # Add progress indicator at the top for multi-page documents
                        if num_pages > 1:
                            progress_header = f"📄 **Document Conversion Progress** `[Request {request_id}]` (Processing page {min(current_page, num_pages)} of {num_pages})\n\n"
                            yield progress_header + process_tags(markdown_content)
                        else:
                            yield process_tags(markdown_content)

                        # Estimate current page based on content length (rough approximation)
                        if "---" in markdown_content:
                            current_page = markdown_content.count("---") + 1

                        # Reduced delay for better concurrent performance
                        time.sleep(0.01)

                except Exception as e:
                    error_message = f"❌ **Error processing request {request_id}**: {str(e)}\n\nPlease try again or contact support if the issue persists."
                    yield error_message

            # Enable concurrent request processing by setting concurrency_limit
            # This allows multiple users to process documents simultaneously
            submit_btn.click(
                process_markdown_streaming,
                inputs=[images_input],
                outputs=[formatted_output],
                concurrency_limit=concurrency_limit,  # Allow multiple concurrent requests
                concurrency_id="pdf_to_markdown_conversion",  # Unique ID for this processing pipeline
            )
