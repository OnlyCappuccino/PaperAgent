import redis
from rq import Worker
from app.config import get_settings

if __name__ == "__main__":
    conn = redis.Redis.from_url(get_settings().redis_url)
    worker = Worker(["eval"], connection=conn)
    worker.work()
