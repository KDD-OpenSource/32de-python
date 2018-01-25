FROM ubuntu:17.10
EXPOSE 8000
# TODO: Do we really need python3-dev?
RUN apt-get update && apt-get install -y python3-pip python3-dev git
# TODO: Remove -b when on master
# TODO: Force docker to pull each time
RUN git clone -b configure-docker https://github.com/KDD-OpenSource/32de-python.git
WORKDIR 32de-python
RUN pip3 install -r requirements.txt
RUN pip3 install -r deployment.txt
ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:8000", "api.server:app"]