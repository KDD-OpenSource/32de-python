docker build -t server -f Dockerfile .
docker run -d --name server-container server
docker ps -a
docker exec server-container pytest
docker rm -f server-container