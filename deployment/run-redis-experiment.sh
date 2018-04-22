#!/usr/bin/env bash
docker stop redis-meta-paths-experiment
docker rm redis-meta-paths-experiment
docker run --name redis-meta-paths-experiment -p 6380:6380 -d redis
