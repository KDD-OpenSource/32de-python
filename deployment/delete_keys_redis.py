from util.config import REDIS_PASSWORD, REDIS_PORT, REDIS_HOST
import redis

if __name__=='__main__':
    client = redis.StrictRedis(host=REDIS_HOST, password=REDIS_PASSWORD, port=REDIS_PORT)
    print(client.delete(*client.keys()))