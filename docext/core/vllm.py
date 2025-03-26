from __future__ import annotations

import os
import signal
import subprocess
import threading
import time

import requests


class VLLMServer:
    def __init__(self, model_name, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.server_process = None
        self.url = f"http://{self.host}:{self.port}/v1/models"

    def start_server(self):
        """Start the vLLM server in a background thread."""
        print("Starting vLLM server...")
        # Command to start the vLLM server
        # vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=1,video=0 --tensor-parallel-size 1 --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 20000 --gpu-memory-utilization 0.95 --enforce-eager;
        command = [
            "vllm",
            "serve",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            "bfloat16",
            "--limit-mm-per-prompt",
            "image=5,video=0",
            "--served-model-name",
            self.model_name,
            "--max-model-len",
            "20000",
            "--gpu-memory-utilization",
            "0.95",
            "--enforce-eager",
        ]

        # Start the server as a subprocess
        self.server_process = subprocess.Popen(command)
        # self.server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def wait_for_server(self, timeout=300):
        """Wait until the vLLM server is ready."""
        print("Waiting for vLLM server to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.url)
                if response.status_code == 200:
                    print(
                        f"vLLM server started on {self.host}:{self.port} with PID: {self.server_process.pid}",
                    )
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)

        print("Error: vLLM server did not start in time.")
        self.stop_server()
        exit(1)

    def stop_server(self):
        """Stop the vLLM server gracefully."""
        if self.server_process:
            print("Stopping vLLM server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("vLLM server stopped.")

    def run_in_background(self):
        """Run the server in a background thread and wait for readiness."""
        server_thread = threading.Thread(target=self.start_server, daemon=True)
        server_thread.start()
        self.wait_for_server()
        return server_thread
