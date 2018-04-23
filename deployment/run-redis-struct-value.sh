#!/usr/bin/env bash
docker stop redis-meta-paths-struct-value
docker rm redis-meta-paths-struct-value
docker run --name redis-meta-paths-struct-value -p 6381:6379 -d redis
