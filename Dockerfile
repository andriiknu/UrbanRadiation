FROM ubuntu:16.04

## Install General Requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    git \
    wget \
    nano \
    python3-pip \
    python-pip \
    python3-dev \
    python-dev \
    software-properties-common

# Install setuptools using apt
RUN apt-get install -y python-setuptools python3-setuptools

# Install a specific version of pip that is compatible with the current environment
RUN pip install pip==20.3.4
RUN pip3 install pip==20.3.4

# Upgrade pip for both Python 2 and Python 3 to the latest version
RUN pip install --upgrade pip
RUN pip3 install --upgrade pip

# Install other Python packages
RUN pip install --ignore-installed six
RUN pip install numpy==1.15.4
RUN pip install setuptools==39.1.0
RUN pip install tensorflow==1.10.0
RUN pip install Keras==2.2.4
RUN pip install scipy==1.1.0
RUN pip install joblib==0.13.2
RUN pip install matplotlib==2.2.3
RUN pip install tqdm==4.23.4
RUN pip install pandas==0.24.2
RUN pip install scikit-learn==0.20.3
RUN pip install h5py==2.9.0
RUN pip install gdown

WORKDIR /work