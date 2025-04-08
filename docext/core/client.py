from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from litellm import completion

load_dotenv()


def sync_request(
    messages: list[dict],
    model_name: str = "hosted_vllm/Qwen/Qwen2.5-VL-3B-Instruct",
    max_tokens: int = 5000,
    num_completions: int = 1,
):
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1/")
    response = completion(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        n=num_completions,
        api_base=vllm_url if model_name.startswith("hosted_vllm/") else None,
    )
    return response.json()
