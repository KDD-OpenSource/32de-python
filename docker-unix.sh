#!/bin/bash
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --volume=$HOME/neo4j/logs:/logs \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    --rm \
    -d \
    neo4j:3.3-enterprise

# How to integrate our neo4j code changes (meta-path computation)?

git clone -b configure-docker https://github.com/KDD-OpenSource/32de-python.git
cd 32de-python/
docker build -t server -f Dockerfile .
docker run server --name server-container --publish=8000:8000 --rm -d
cd ..

git clone https://github.com/KDD-OpenSource/32de-UI.git
cd 32de-UI/
docker build -t ui -f Dockerfile .
docker run ui --name ui-container --publish=80:80 --rm -d
cd ..