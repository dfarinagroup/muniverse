FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge
WORKDIR /opt
RUN wget https://github.com/conda-forge/miniforge/releases/download/23.1.0-2/Mambaforge-Linux-x86_64.sh && \
    bash Mambaforge-Linux-x86_64.sh -b -p /opt/mambaforge && \
    rm Mambaforge-Linux-x86_64.sh
ENV PATH=/opt/mambaforge/bin:$PATH

# Clone the scd repo
RUN git clone https://github.com/AgneGris/swarm-contrastive-decomposition.git scd

# Switch to repo directory
WORKDIR /opt/scd

# Create the conda environment
RUN mamba env create -f decomposition.yml && \
    mamba clean --all -f -y && \
    rm -rf ~/.cache ~/.conda

# Install the package in development mode
RUN source /opt/mambaforge/etc/profile.d/conda.sh && \
    conda activate decomposition && \
    pip install -e .

CMD ["/bin/bash"]
