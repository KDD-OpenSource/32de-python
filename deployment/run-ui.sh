#!/bin/bash
docker stop ui-container
docker rm ui-container
docker run --name ui-container \
           --publish=80:80 \
           -d \
           ui
