FROM condaforge/mambaforge:latest

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt

# Clone the NeuroMotion repo
RUN git clone https://github.com/pranavm19/NeuroMotion

# Switch to repo directory
WORKDIR /opt/NeuroMotion

# Create the conda environment
RUN mamba env create -f env.yml && \
    mamba clean --all -f -y && \
    rm -rf ~/.cache ~/.conda

# Install extra dependencies into the conda env
RUN mamba run -n NeuroMotion pip install \
    git+https://github.com/shihan-ma/BioMime.git \
    gdown

RUN mamba run -n NeuroMotion bash get_data.sh

CMD ["/bin/bash"]
