#!/bin/bash
#mkdir datasets

xhost +local:
docker run -it --net=host \
  --volume=/dev:/dev \
  --name=dockerdoordetector \
  --workdir=/home/$USER \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e CONTAINER_NAME=dockerdoordetector-dev \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$(pwd)/..:/home/$USER" \
  dockerdoordetector:latest
#  -v "/home/markpp/datasets:/home/markpp/datasets" \
