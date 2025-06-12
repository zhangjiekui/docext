from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocExt: Onprem information extraction from documents",
    )
    parser.add_argument(
        "--vlm_server_port",
        type=int,
        default=8000,
        help="Port for the vLLM/OLLAMA server",
    )
    parser.add_argument(
        "--vlm_server_host",
        type=str,
        default="127.0.0.1",
        help="Host for the vLLM/OLLAMA server",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        help="Name of the model to use. Use 'ollama/' prefix for OLLAMA models and 'hosted_vllm/' prefix for hosted vLLM models.",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        dest="ui_port",
        default=7860,
        help="Port for the gradio UI",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=15000,
        help="Maximum length of the model. Use small values for low memory devices.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization. Use a value between 0 and 1.",
    )
    parser.add_argument(
        "--max_num_imgs",
        type=int,
        default=5,
        help="Maximum number of images to process in a single prompt.",
    )
    parser.add_argument(
        "--vllm_start_timeout",
        type=int,
        default=300,
        help="Timeout for the vLLM server to start.",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        dest="share",  # This will set 'share' to False when --no-share is used
        help="Disable sharing of the UI on the web.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        help="Log level. Can be 'debug', 'info', 'warning', 'error', 'critical'.",
    )
    parser.add_argument(
        "--max_img_size",
        type=int,
        default=2048,
        help="Maximum size of the image to process. Use 1024 for low memory devices.",
    )
    parser.add_argument(
        "--concurrency_limit",
        type=int,
        default=1,
        help="Maximum number of concurrent PDF to markdown conversion requests. Higher values allow more users to process documents simultaneously but require more memory and compute resources.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for the model. Can be 'bfloat16' or 'float16'.",
    )
    parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens to generate for the model.",
    )
    return parser.parse_args()
