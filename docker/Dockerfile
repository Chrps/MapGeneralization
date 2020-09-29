FROM ubuntu:18.04

LABEL maintainer "Mark - Auto generated"

ENV DEBIAN_FRONTEND=noninteractive

# Install standard tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils software-properties-common dirmngr build-essential \ 
    gpg-agent lsb-release git curl unzip wget gedit nano cmake \
    pkg-config checkinstall yasm libopenblas-dev 

# install python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-tk python3-setuptools

# Update pip

RUN pip3 --no-cache-dir install --upgrade \
    pip setuptools

# Install pytorch
RUN pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html 

# RUN pip3 install http://download.pytorch.org/whl/cpu/torch-1.6.0-cp36-cp36m-manylinux1_x86_64.whl

RUN pip3 install torchvision

# Install Deep Graph Library (DGL)
RUN pip3 install dgl==0.5.1
ENV DGLBACKEND=pytorch

# Install pytorch-lightning
RUN pip3 install pytorch-lightning

# Install misc
RUN pip3 install sklearn gensim pyyaml

RUN pip3 install matplotlib

# RUN rm -rf /var/lib/apt/lists/*

# COPY user.sh /usr/local/bin/
# CMD bash -C '/usr/local/bin/user.sh'
