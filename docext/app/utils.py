from __future__ import annotations


def cleanup(signum, frame, vllm_server):
    print("\nReceived exit signal. Stopping vLLM server...")
    vllm_server.stop_server()
    exit(0)
