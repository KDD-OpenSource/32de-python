FROM ubuntu:17.10
EXPOSE 8000
# TODO: Do we really need python3-dev?
RUN apt-get update && apt-get install -y python3-pip python3-dev git dirmngr
# TODO: Remove -b when on master
# TODO: Force docker to pull each time

RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25
RUN echo "deb http://downloads.skewed.de/apt/xenial xenial universe" | tee -a /etc/apt/sources.list
RUN echo "deb-src http://downloads.skewed.de/apt/xenial xenial universe" | tee -a /etc/apt/sources.list
# RUN  apt-get install boost coal cairomm python-cairo
RUN apt-get update -qq && apt-get install python3-graph-tool -y

RUN git clone -b configure-docker https://github.com/KDD-OpenSource/32de-python.git
WORKDIR 32de-python
RUN pip3 install -r requirements.txt
RUN pip3 install -r deployment.txt
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "api.server:app"]