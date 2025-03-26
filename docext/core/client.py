from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def sync_request(
    messages: list[dict],
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_tokens: int = 5000,
):
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
    response = requests.post(
        vllm_url,
        json={
            "model": model_name,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": False,
        },
        headers={"Content-Type": "application/json"},
    )
    return response.json()
