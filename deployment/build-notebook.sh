#!/bin/bash
docker build -t notebook -f Dockerfile-Notebook ${1:-.}
