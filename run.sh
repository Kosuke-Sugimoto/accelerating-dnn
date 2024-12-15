#!/bin/bash

DOCKER_IMAGE_NAME=adnn:latest
DOCKER_CONTAINER_NAME=adnn
CURRENT_WORKING_DIR=$(pwd)/

if [[ "$(docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == ""  ]]
then
    docker build -t $DOCKER_IMAGE_NAME .
fi

docker run -it \
           --name $DOCKER_CONTAINER_NAME \
           --ipc=host \
           --gpus all \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --mount type=bind,src=$CURRENT_WORKING_DIR,dst=/work/ \
           $DOCKER_IMAGE_NAME
