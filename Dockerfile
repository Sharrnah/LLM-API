#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    git \
    git-lfs \
    build-essential \
    gcc \
    libglib2.0-0 \
    wget \
    llvm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# install debian python package
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    python3.10 \
    python3-dev \
    python3-pip \
    libffi-dev libnacl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
#    rm ~/miniconda.sh
#ENV PATH=$CONDA_DIR/bin:$PATH
#
## install python 3.10 and pip
#RUN conda install python=3.10 && conda install -c anaconda pip
#
#
#RUN mkdir -p /root/.huggingface/
#
#RUN conda install -c "nvidia/label/cuda-12.2.2" cuda-toolkit
#RUN conda install pytorch torchvision torchaudio -c pytorch -c conda-forge

RUN pip install --no-cache-dir --upgrade pip
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ framework selection ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
##################
# llama-cpp cuda #
##################
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
RUN apt-get update -q && apt install -y ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev \
    libopenblas-dev \
    && apt-get clean
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
################
# llama-cpp CL #
################
#ENV CMAKE_ARGS="-DLLAMA_CLBLAST=on"
##################
# llama-OpenBLAS #
##################
#ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
#RUN apt update && apt install -y libopenblas-dev ninja-build
#RUN pip install --no-cache-dir --upgrade pytest cmake scikit-build setuptools pydantic-settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ /framework selection ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WORKDIR /app
EXPOSE 8000

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -U
RUN test -d /root/.cache || mkdir -p /root/.cache
COPY cache/. /root/.cache/

COPY _preinstall.py /app/_preinstall.py
RUN python3 _preinstall.py

COPY main.py /app/main.py
COPY chatbot.py /app/chatbot.py
COPY chat_history.py /app/chat_history.py
COPY summary_generator.py /app/summary_generator.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
