[![Build Status](https://travis-ci.org/KDD-OpenSource/32de-python.svg?branch=master)](https://travis-ci.org/KDD-OpenSource/32de-python)
[![Coverage Status](https://coveralls.io/repos/github/KDD-OpenSource/32de-python/badge.svg?branch=master)](https://coveralls.io/github/KDD-OpenSource/32de-python?branch=master)

To deploy our system including neo4j, the neo4j graph algorithm component, the UI and our server install docker on your system and run `deployment/docker-deployment.sh`.
This will install a clean version from the alpha-dev and the master branches and doesn't include your local code changes.
To build your own local code use `deployment/build-*.sh /path/to/code` (e.g. `deployment/build-server.sh .`) and to run a single container `deployment/run-*.sh`.
Neo4j browser is listening on port `7474`, bolt is available on port `7687` and our server is listening on port `8000` for all hosts.

Tutorials for installing Docker: [Mac](https://docs.docker.com/docker-for-mac/install/), [Windows](https://docs.docker.com/docker-for-windows/install/) and [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/).
