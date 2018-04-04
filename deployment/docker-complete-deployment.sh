#!/bin/bash

docker rm neo4j-graph-algo-container
docker rm server-container
docker rm ui-container

mkdir tmp/
cd tmp/

########### 32de-python
git clone -b configure-docker https://github.com/KDD-OpenSource/32de-python.git
cd 32de-python/
deployment/build-server.sh .
deployment/run-server.sh

cd ..

########### neo4j-graph-algorithms
git clone -b alpha-dev https://github.com/KDD-OpenSource/neo4j-graph-algorithms.git
cd neo4j-graph-algorithms/

../32de-python/deployment/build-neo4j.sh .
../32de-python/deployment/run-neo4j.sh
cd ..

########### 32de-UI
git clone https://github.com/KDD-OpenSource/32de-UI.git
cd 32de-UI/

../32de-python/deployment/build-ui.sh . https://hpi.de/mueller/metaexp-demo-api/
../32de-python/deployment/run-ui.sh

cd ..

read -p "Press enter to terminate the containers"
cd ..
rm -rf tmp/

docker stop neo4j-graph-algo-container
docker stop server-container
docker stop ui-container

docker rm neo4j-graph-algo-container
docker rm server-container
docker rm ui-container