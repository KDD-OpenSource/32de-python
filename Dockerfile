FROM ubuntu:16.04
EXPOSE 8000
# TODO: Do we really need python3-dev?
RUN apt-get update && apt-get install -y python3-pip python3 dirmngr

RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25
RUN echo "deb http://downloads.skewed.de/apt/xenial xenial universe" | tee -a /etc/apt/sources.list
RUN echo "deb-src http://downloads.skewed.de/apt/xenial xenial universe" | tee -a /etc/apt/sources.list
RUN apt-get update && apt-get install -y libboost-all-dev
RUN apt-get update -qq && apt-get install -y python3-graph-tool

# Please add dependencies here and in requirements.txt or deployment/deployment.txt
RUN pip3 install numpy==1.14.0 redis sklearn typing pytest flask pandas flask-cors scipy pytest-cov Flask-Session graphviz cryptography==2.1.4 flask-ask
RUN pip3 install gunicorn

COPY . /32de-python/

WORKDIR /32de-python
ENTRYPOINT ["gunicorn", "--config", "deployment/gunicorn-config.py", "api.server:app"]