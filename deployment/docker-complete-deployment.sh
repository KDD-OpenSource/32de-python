#!/bin/bash

docker rm neo4j-graph-algo-container
docker rm server-container
docker rm ui-production-container
docker rm ui-dev-container

mkdir tmp/
cd tmp/

########### 32de-python
git clone https://github.com/KDD-OpenSource/32de-python.git
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

export REACT_APP_API_HOST=${1:http://localhost:8000/}
deployment/build-dev-ui.sh .
deployment/build-production-ui.sh . ${1:http://localhost:8000/}
deployment/run-production-ui.sh

cd ..

read -p "Press enter to terminate the containers"
cd ..
#rm -rf tmp/

docker stop neo4j-graph-algo-container
docker stop server-container
docker stop ui-production-container
docker stop ui-dev-container

#docker rm neo4j-graph-algo-container
#docker rm server-container
#docker rm ui-production-container
#docker rm ui-dev-container
