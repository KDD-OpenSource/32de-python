#!/usr/bin/env bash
docker cp notebook-container:/32de-python/notebooks/. ${1:-.}/notebooks/.
