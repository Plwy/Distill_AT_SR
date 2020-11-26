FROM nvidia/cuda:10.2-base-ubuntu16.04

ENV LANG en_US.UTF-8 
ENV LC_ALL en_US.UTF-8

RUN apt-get update -y && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update \
 && apt-get install -y curl unzip \
    python3.6 python3-pip python3-setuptools \
 && ln -s /usr/bin/python3.6 /usr/bin/python \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

COPY requirements.txt /src/

WORKDIR /src
RUN pip install -r /src/requirements.txt

