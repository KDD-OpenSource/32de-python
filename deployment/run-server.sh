#!/bin/bash
docker run --name server-container \
           --publish=8000:8000 \
           -d \
           server