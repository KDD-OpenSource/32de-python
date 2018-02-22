#!/bin/bash
docker stop neo4j-graph-algo-container-commerzbank
docker rm neo4j-graph-algo-container-commerzbank
docker run \
    --publish=7494:7474 --publish=7707:7687 \
    --volume=$HOME/commerzbank/data:/data \
    --volume=$HOME/commerzbank/logs:/logs \
    --volume=$HOME/commerzbank/conf:/conf \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d \
    --name neo4j-graph-algo-container-commerzbank \
    neo4j-graph-algorithms

