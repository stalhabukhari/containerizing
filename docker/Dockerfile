# This Dockerfile sets up a CUDA-enabled environment with JAX
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y \
        wget git net-tools vim curl build-essential x11vnc zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev freeglut3-dev && \
        # following is for opencv-python (uncomment if not needed)
        apt install -y --no-install-recommends libglib2.0-0 libxrender1 libxext6 libsm6 libgl1-mesa-glx && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    echo "Running $(conda --version)"

# install python package requirements
COPY requirements.txt /env-setup/requirements.txt
RUN conda init bash && . /root/.bashrc && conda create -n env_jax python=3.9 && \
    source activate env_jax && pip install --upgrade pip && \
    cd /env-setup && pip install -r requirements.txt

# install jax
RUN source activate env_jax && pip install "jax[cuda12]" && \
    pip install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# path to mount code repository
WORKDIR /code-dir
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env_jax", "/bin/bash", "-c"]