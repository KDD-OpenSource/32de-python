#!/bin/bash
docker cp . server-container:/32de-python/
docker exec server-container pip3 install -r requirements.txt
docker exec server-container pip3 install -r deployment/deployment.txt