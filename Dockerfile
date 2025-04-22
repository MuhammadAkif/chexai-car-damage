# Use Python 3.9 as base image
# FROM python:3.9-slim
FROM ubuntu:22.04
# Set working directory
WORKDIR /app

# Install system dependencies including required packages for OpenCV and lap
# Install system dependencies including required packages for OpenCV, pip, and bash
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    build-essential \
    python3 \
    python3-pip \
    bash \
    && rm -rf /var/lib/apt/lists/*



# Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#     bash miniconda.sh -b -p /opt/conda && \
#     rm miniconda.sh

# # Set PATH
# ENV PATH="/opt/conda/bin:${PATH}"

# # Update conda and install lap
# RUN conda update -n base -c defaults conda && \
#     conda install -y -c conda-forge lap

# Create necessary directories
RUN mkdir -p /app/s3_files /app/AiModels

# Copy requirements files first to leverage Docker cache
COPY requirements.txt requirements_yolov9.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_yolov9.txt

# Set environment variables
ENV PYTHONPATH="/app"
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Copy only necessary project files and directories
COPY AiModels/*.pt ./AiModels/
COPY Auth/ ./Auth/
COPY Model/ ./Model/
COPY models/ ./models/
COPY Router/ ./Router/
COPY Services/ ./Services/
COPY utils/ ./utils/
COPY Utils/ ./Utils/
COPY main.py export.py ./

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]