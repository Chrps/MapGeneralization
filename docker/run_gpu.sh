#!/bin/bash
#mkdir datasets

xhost +local:
docker run -it --net=host \
  --gpus all \
  --volume=/dev:/dev \
  --name=dockermapspeople \
  --workdir=/home/$USER \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e CONTAINER_NAME=dockermapspeople-dev \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$(pwd)/..:/home/$USER" \
  dockermapspeople:latest
#  -v "/home/markpp/datasets:/home/markpp/datasets" \
