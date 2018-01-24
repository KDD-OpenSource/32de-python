FROM ubuntu:17.10
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN pip3 install -r requirements.txt
RUN pip3 install -r deployment.txt
RUN git clone https://github.com/KDD-OpenSource/32de-python.git
RUN cd 32de-python && gunicorn --bind 0.0.0.0:8000 api.server:app