# Use a slim Python 3.11 base image
FROM python:3.11-slim

# Install system dependencies needed for OpenCV, PyTorch, and general builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user and working directory (required for HF Spaces)
# HF Spaces runs containers with user UID 1000
RUN useradd -m -u 1000 user
WORKDIR /code

# Set Hugging Face cache directory to a writeable location
ENV HF_HOME=/code/.cache/huggingface
RUN mkdir -p /code/.cache/huggingface && chown -R user:user /code

# Copy requirements file first to leverage Docker cache
COPY --chown=user:user requirements.txt /code/requirements.txt

# Install dependencies
# Optimize PyTorch installation to use CPU-only binaries to save space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . /code

# Pre-download Hugging Face models during build to speed up container startup.
# This avoids container startup timeouts and lazy-loading delays.
RUN python -c " \
from transformers import pipeline; \
pipeline('image-classification', model='umm-maybe/AI-image-detector'); \
pipeline('image-classification', model='dima806/ai_vs_real_image_detection'); \
pipeline('image-classification', model='Organika/sdxl-detector') \
"

# Set permissions
RUN chmod -R 777 /code

# Switch to non-root user
USER user

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run uvicorn server mapping to port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
