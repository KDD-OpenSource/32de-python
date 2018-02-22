#!/bin/bash
docker stop neo4j-graph-algo-container
docker rm neo4j-graph-algo-container
docker run \
    --publish=7504:7474 --publish=7717:7687 \
    --volume=$HOME/neo4j/data:/data \
    --volume=$HOME/neo4j/logs:/logs \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d \
    --name neo4j-graph-algo-container \
    neo4j-graph-algorithms
