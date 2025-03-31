FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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

# Clone the NeuroMotion repo with specific version
RUN git clone https://github.com/shihan-ma/NeuroMotion.git && \
    cd NeuroMotion && \
    git checkout 590369ec2f395e6e228aa9dc58bf0fb87a2c0329

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

# Create directories with proper permissions
RUN mkdir -p ckp && \
    mkdir -p NeuroMotion/MSKlib/models

# Download the files using gdown with error checking
RUN mamba run -n NeuroMotion bash -c '\
    for url in "1RIYnYxLkBZ9_7MJQgQBSjAk_oXBTqY0b" "1xxeXF4RS7qH-kq3yrfCiWIQ7DemMounr" "10PoLwJcHoS8ZRvqlExENpt3tt2Vus72s" "1HD3UQFgaxSTsZIFW04bp-pLmn8UROBEp"; do \
        if ! gdown $url; then \
            echo "Failed to download file for URL: $url" && exit 1; \
        fi \
    done'

RUN mamba run -n NeuroMotion bash -c '\
    gdown 1RIYnYxLkBZ9_7MJQgQBSjAk_oXBTqY0b -O ckp/model_linear.pth && \
    gdown 1xxeXF4RS7qH-kq3yrfCiWIQ7DemMounr -O ckp/muap_example.pkl && \
    gdown 10PoLwJcHoS8ZRvqlExENpt3tt2Vus72s -O NeuroMotion/MSKlib/models/poses.csv && \
    gdown 1HD3UQFgaxSTsZIFW04bp-pLmn8UROBEp -O NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3.zip'

RUN unzip NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3.zip -d NeuroMotion/MSKlib/models/ && \
    rm NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3.zip


CMD ["/bin/bash"]
