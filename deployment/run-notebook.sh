#!/bin/bash
docker stop notebook-container
docker rm notebook-container
docker run --name notebook-container \
           --publish=${1:-8888}:8888 \
           -d \
           notebook