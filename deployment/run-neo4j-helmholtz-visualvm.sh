#!/bin/bash 
docker stop neo4j-graph-algo-container-helmholtz
docker rm neo4j-graph-algo-container-helmholtz
docker run \
    --publish=7484:7474 --publish=7697:7687 --publish=3637:3637 --publish=1099:1099 \
    --volume=$HOME/bioDB/data:/data \
    --volume=$HOME/bioDB/logs:/logs \
    --volume=$HOME/bioDB/conf:/conf \
    --env=NEO4J_dbms_logs_query_enabled=true \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    --env=NEO4J_dbms_transaction_timeout=40s \
    --env=NEO4J_dbms_memory_pagecache_size=40G \
    --env=NEO4J_dbms_memory_heap_initial__size=50G \
    --env=NEO4J_dbms_memory_heap_max__size=200G \
    --env=-Dcom.sun.management.jmxremote.port=3637 \
    --env=-Dcom.sun.management.jmxremote.local.only=false \
    --env=-Dcom.sun.management.jmxremote.authenticate=false \
    --env=-Dcom.sun.management.jmxremote.ssl=false \
    --env=NEO4J_AUTH=neo4j/${HELMHOLTZ_PWD} \
    -d \
    --name neo4j-graph-algo-container-helmholtz \
    neo4j-graph-algorithms

