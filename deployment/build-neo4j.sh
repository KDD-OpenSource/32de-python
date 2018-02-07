#!/bin/bash
# Call this script with the path of the context (your code) which should be included in the docker container
# $1 is the first command line argument
docker build -t neo4j-graph-algorithms -f Dockerfile $1