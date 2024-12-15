FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pythonコマンドを通すための設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --set python /usr/bin/python3.10

# pipのアップグレード
RUN python -m pip install --upgrade pip

WORKDIR /work
