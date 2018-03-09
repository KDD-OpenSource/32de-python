#!/bin/bash
docker build -t notebook -f Dockerfile.jupyter ${1:-.}