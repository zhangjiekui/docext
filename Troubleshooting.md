## Troubleshooting

### 1. `ValueError: Could not fetch api info for https://e50b2cd4028ae78975.gradio.live/: {"detail":"Not Found"}`
- Ensure the client is using Gradio version `5.23.2`.

### 2. Out of Memory Error
- Try reducing the following parameters:
  - `max_num_imgs`
  - `max_model_len`
  - `max_img_size`
- If possible, switch to a GPU with more memory.

### 3. `RuntimeError: Failed to infer device type`
- This error occurs when CUDA drivers are not installed, affecting vLLM.
- Follow the troubleshooting guide [here](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#failed-to-infer-device-type).

### 4. `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your Tesla T4 GPU has compute capability 7.5. You can use float16 instead by explicitly setting the `dtype` flag in CLI, for example: --dtype=half.`
- Use `--dtype=float16` instead of `--dtype=bfloat16`.
