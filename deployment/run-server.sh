#!/bin/bash
docker stop server-container
docker rm server-container
docker run --name server-container \
           --publish=8000:8000 \
           -d \
           server