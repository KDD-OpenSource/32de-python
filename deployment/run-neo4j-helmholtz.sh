#!/bin/bash
docker stop neo4j-graph-algo-container-helmholtz
docker rm neo4j-graph-algo-container-helmholtz
docker run \
    --publish=7484:7474 --publish=7697:7687 \
    --volume=$HOME/bioDB/data:/data \
    --volume=$HOME/bioDB/logs:/logs \
    --volume=$HOME/bioDB/conf:/conf \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d \
    --name neo4j-graph-algo-container-helmholtz \
    neo4j-graph-algorithms

