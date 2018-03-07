#!/bin/bash
docker stop neo4j-graph-algo-container-commerzbank
docker rm neo4j-graph-algo-container-commerzbank
docker run \
    --publish=7494:7474 --publish=7707:7687 --publish=3636:3636\
    --volume=$HOME/commerzbank/data:/data \
    --volume=$HOME/commerzbank/logs:/logs \
    --volume=$HOME/commerzbank/conf:/conf \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    --env=NEO4J_dbms_memory_pagecache_size=40G \
    --env=NEO4J_dbms_memory_heap_initial__size=50G \
    --env=NEO4J_dbms_memory_heap_max__size=200G \
    --env=-Djava.rmi.server.hostname=172.20.14.22 \
    --env=-Dcom.sun.management.jmxremote.rmi.port=3636 \
    --env=-Dcom.sun.management.jmxremote \
    --env=-Dcom.sun.management.jmxremote.port=3636 \
    --env=-Dcom.sun.management.jmxremote.local.only=false \
    --env=-Dcom.sun.management.jmxremote.authenticate=false \
    --env=-Dcom.sun.management.jmxremote.ssl=false \
    --env=NEO4J_AUTH=neo4j/${COMMERZBANK_PWD} \
    -d \
    --name neo4j-graph-algo-container-commerzbank \
    neo4j-graph-algorithms

