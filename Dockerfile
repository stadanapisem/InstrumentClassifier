FROM nvidia/cuda:9.1-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    xvfb \
    fluxbox \
    x11vnc \
    gcc \
    g++ \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.4 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Ensure conda version is at least 4.4.11
# (because of this issue: https://github.com/conda/conda/issues/6811)
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda install -y "conda>=4.4.11" && conda clean -ya

# CUDA 9.1-specific steps
RUN conda install -y -c pytorch \
    cuda91=1.0 \
    magma-cuda91=2.3.0 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y \
    h5py \
 && conda clean -ya
RUN pip install --upgrade pip
RUN pip install h5py-cache

RUN conda install -c conda-forge tqdm
RUN conda install pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install -c mingfeima mkldnn

# Install Graphviz
RUN conda install -y graphviz=2.38.0 \
 && conda clean -ya

RUN pip install numpy scipy matplotlib python_speech_features dill
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN git clone --recursive https://github.com/pytorch/pytorch
RUN cd pytorch && python setup.py install
