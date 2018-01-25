#!/bin/bash
mkdir tmp/
cd tmp/

# TODO: Remove -b when on master
git clone -b configure-docker https://github.com/KDD-OpenSource/32de-python.git
cd 32de-python/

# TODO: Is there a nicer way than --no-cache? Maybe clone outside
docker build --no-cache -t neo4j-graph-algo -f Dockerfile-neo4j-graph-algorithms .
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --volume=$HOME/neo4j/logs:/logs \
    --volume=$HOME/neo4j/plugins:/plugins \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d \
    --name neo4j-graph-algo-container \
    neo4j-graph-algo

docker build --no-cache -t server -f Dockerfile .
docker run --name server-container \
           --publish=8000:8000 \
           -d \
           server
cd ..

git clone https://github.com/KDD-OpenSource/32de-UI.git
cd 32de-UI/
docker build --no-cache -t ui -f Dockerfile .
docker run --name ui-container \
           --publish=80:80 \
           -d \
           ui
cd ..
read -p "Press enter to terminate the containers"
cd ..
rm -rf tmp/