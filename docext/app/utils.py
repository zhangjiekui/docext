from __future__ import annotations

from PIL import Image


def cleanup(signum, frame, vllm_server):
    print("\nReceived exit signal. Stopping vLLM server...")
    vllm_server.stop_server()
    exit(0)
