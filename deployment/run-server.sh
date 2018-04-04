#!/bin/bash
docker stop server-container
docker rm server-container
docker run --name server-container \
           --publish=${1:-8000}:8000 \
           -d \
	       -e "METAEXP_DEV=true"\
           server
