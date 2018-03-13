[![Build Status](https://travis-ci.org/KDD-OpenSource/32de-python.svg?branch=master)](https://travis-ci.org/KDD-OpenSource/32de-python)
[![Coverage Status](https://coveralls.io/repos/github/KDD-OpenSource/32de-python/badge.svg?branch=master)](https://coveralls.io/github/KDD-OpenSource/32de-python?branch=master)

# Deployment
To deploy our system including neo4j, the neo4j graph algorithm component, the UI and our server install docker on your system and run `deployment/docker-deployment.sh`.
This will install a clean version from the alpha-dev and the master branches and doesn't include your local code changes.
If the API should be served ssl encrypted, set the environment variable `METAEXP_HTTPS` to `true` and provide `api.crt` and `api.key` in the `https` folder.

# Development
To build your own local code use `deployment/build-*.sh /path/to/code` (e.g. `deployment/build-server.sh .`) and to run a single container `deployment/run-*.sh`.
Neo4j browser is listening on port `7474`, bolt is available on port `7687` and our server is listening on port `8000` for all hosts. 
If you start the additional neo4j containers with `run-neo4j-helmholtz.sh` and `run-neo4j-commerzbank.sh`, they are listening on the ports +10 for Helmholtz and +20 for the Commerzbank data.
All the neo4j containers are based on the `neo4j-graph-algorithms` image.

Tutorials for installing Docker: [Mac](https://docs.docker.com/docker-for-mac/install/), [Windows](https://docs.docker.com/docker-for-windows/install/) and [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/).
