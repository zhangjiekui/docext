FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev

# Install Python 3.11 and pip
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip python3.11-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.11 is default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && ln -sf /usr/bin/python3.11 /usr/bin/python3

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app

# Activate virtual environment and install dependencies
COPY requirements.txt setup.py README.md /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY docext /app/docext

# Install application
RUN pip install --no-cache-dir -e .

# Install flash-attn separately
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Set working directory and entrypoint
WORKDIR /app/docext
ENTRYPOINT ["python", "-m", "docext.app.app", "--no-share", "--ui_port", "7860"]
