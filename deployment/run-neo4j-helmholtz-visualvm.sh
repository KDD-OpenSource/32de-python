#!/bin/bash
docker stop neo4j-graph-algo-container-helmholtz
docker rm neo4j-graph-algo-container-helmholtz
docker run \
    --publish=7484:7474 --publish=7697:7687 --publish=3637:3637 --publish=1099:1099 \
    --volume=$HOME/bioDB/data:/data \
    --volume=$HOME/bioDB/logs:/logs \
    --volume=$HOME/bioDB/conf:/conf \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    --env=-Djava.rmi.server.hostname=172.20.14.22 \
    --env=-Dcom.sun.management.jmxremote.rmi.port=3637 \
    --env=-Dcom.sun.management.jmxremote \
    --env=-Dcom.sun.management.jmxremote.port=3637 \
    --env=-Dcom.sun.management.jmxremote.local.only=false \
    --env=-Dcom.sun.management.jmxremote.authenticate=false \
    --env=-Dcom.sun.management.jmxremote.ssl=false \
    --env=NEO4J_AUTH=neo4j/${HELMHOLTZ_PWD} \
    -d \
    --name neo4j-graph-algo-container-helmholtz \
    neo4j-graph-algorithms

