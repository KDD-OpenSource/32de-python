#!/bin/bash
# TODO: Delete old files before?
docker cp . server-container:/32de-python/
# TODO: Uninstall old packages?
docker exec server-container pip3 install -r requirements.txt
docker exec server-container pip3 install -r deployment/deployment.txt