#!/bin/bash
docker run --name ui-container \
           --publish=80:80 \
           -d \
           ui