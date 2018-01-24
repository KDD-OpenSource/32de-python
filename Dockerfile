FROM ubuntu:17.10
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN git clone https://github.com/KDD-OpenSource/32de-python.git
WORKDIR 32de-python
RUN pip3 install -r requirements.txt
RUN pip3 install -r deployment.txt
RUN gunicorn --bind 0.0.0.0:8000 api.server:app