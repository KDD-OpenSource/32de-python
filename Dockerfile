FROM ubuntu:16.04
EXPOSE 8000
# TODO: Do we really need python3-dev?
RUN apt-get update && apt-get install -y python3-pip python3 dirmngr

RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25
RUN echo "deb http://downloads.skewed.de/apt/xenial xenial universe" | tee -a /etc/apt/sources.list
RUN echo "deb-src http://downloads.skewed.de/apt/xenial xenial universe" | tee -a /etc/apt/sources.list
RUN apt-get update && apt-get install -y libboost-all-dev
RUN apt-get update -qq && apt-get install -y python3-graph-tool

COPY . /src/

WORKDIR /src
RUN pip3 install -r requirements.txt
RUN pip3 install -r deployment/deployment.txt
ENTRYPOINT ["gunicorn", "--config", "deployment/gunicorn-config.py", "api.server:app"]