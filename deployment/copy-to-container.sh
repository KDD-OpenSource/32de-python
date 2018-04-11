#!/bin/bash
docker cp . ${1:-server-container}:${2:-/32de-python/}
docker exec ${1:-server-container} pip3 install -r ${2:-/32de-python/}requirements.txt
docker exec ${1:-server-container} pip3 install -r ${2:-/32de-python/}deployment/deployment.txt
