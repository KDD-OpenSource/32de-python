#!/bin/bash
docker rm neo4j-graph-algo-container
docker rm server-container
docker rm ui-container

mkdir tmp/
cd tmp/


########### neo4j-graph-algorithms
git clone -b alpha-dev https://github.com/KDD-OpenSource/neo4j-graph-algorithms.git
cd neo4j-graph-algorithms/

docker build -t neo4j-graph-algorithms -f Dockerfile .
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --volume=$HOME/neo4j/logs:/logs \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d \
    --name neo4j-graph-algo-container \
    neo4j-graph-algorithms
cd ..

########### 32de-python
git clone https://github.com/KDD-OpenSource/32de-python.git
cd 32de-python/

docker build -t server -f Dockerfile .
docker run --name server-container \
           --publish=8000:8000 \
           -d \
           server
cd ..

########### 32de-UI
git clone https://github.com/KDD-OpenSource/32de-UI.git
cd 32de-UI/

docker build -t ui -f Dockerfile .
docker run --name ui-container \
           --publish=80:80 \
           -d \
           ui
cd ..
read -p "Press enter to terminate the containers"
cd ..
rm -rf tmp/
docker rm neo4j-graph-algo-container
docker rm server-container
docker rm ui-container