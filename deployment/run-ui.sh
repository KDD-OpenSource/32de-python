#!/bin/bash
docker stop ui-container
docker rm ui-container
docker run --name ui-container \
           --publish=${1:-80}:80 \
           -d \
           ui
