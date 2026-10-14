# Base image with CUDA and development tools
FROM nvidia/cuda:12.9.0-base-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential Linux packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh &&     /bin/bash ~/miniconda.sh -b -p /opt/conda &&     rm ~/miniconda.sh &&     /opt/conda/bin/conda init &&     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&     echo "conda activate base" >> ~/.bashrc

# Set PATH to include conda
ENV PATH /opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n video_chatgpt python=3.10 -y
SHELL ["conda", "run", "-n", "video_chatgpt", "/bin/bash", "-c"]

# Set up the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install FlashAttention
RUN apt-get update && apt-get install -y --no-install-recommends ninja-build && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/HazyResearch/flash-attention.git &&     cd flash-attention &&     git checkout v1.0.7 &&     python setup.py install &&     cd .. &&     rm -rf flash-attention

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose default Gradio port (if the demo uses it)
EXPOSE 7860

# Default command (can be overridden)
CMD ["/bin/bash"]
