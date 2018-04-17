#!/usr/bin/env bash
docker stop redis-meta-paths
docker rm redis-meta-paths
docker run --name redis-meta-paths -p 6379:6379 -v redisData:/data -d redis