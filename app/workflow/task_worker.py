"""
    Task执行器
"""


import json
import logging
import time
from uuid import uuid4

import redis
from rq import Queue
from app.config import get_settings
from app.workflow.value import Evaluator

logger = logging.getLogger(__name__)

def run_eval_task(task_id: str, path: str | None = None, k: int = 5, limit: int | None = None) -> None:
        settings = get_settings()
        redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

        redis_client.hset(TaskWorker._get_task_key(task_id), mapping={
            "status": "running",
            "progress": "0",
            "created_at": str(time.time()),
        })
        try:
            def on_progress(done: int, total: int):
                p = int(done * 100 / total) if total else 0
                redis_client.hset(TaskWorker._get_task_key(task_id), mapping={"progress": str(p), "updated_at": str(time.time())})

            result = Evaluator().evaluate_system(path=path, k=k) if limit is None \
                else Evaluator().evaluate(path=path or "data/eval/sample_eval_questions.jsonl", k=k, limit=limit, progress_cb=on_progress)

            redis_client.hset(TaskWorker._get_task_key(task_id), mapping={
                "status": "succeeded",
                "progress": "100",
                "result": result.model_dump_json(),
                "updated_at": str(time.time()),
            })
        except Exception as e:
            redis_client.hset(TaskWorker._get_task_key(task_id), mapping={
                "status": "failed",
                "error": str(e),
                "updated_at": str(time.time()),
            })
            logger.exception("eval task failed: task_id=%s", task_id)

class TaskWorker():
    def __init__(self):
        self.setting = get_settings()
        self.redis = redis.Redis.from_url(
            self.setting.redis_url,
            decode_responses=True,
        )

    @staticmethod
    def _get_task_key(task_id: str) -> str:
        return f'task:{task_id}'
    

    
    def enqueue_eval_task(self, path: str | None = None, k: int = 5, limit: int | None = None) -> str:
        task_id = str(uuid4())
        self.redis.hset(self._get_task_key(task_id), mapping={
            "status": "queued",
            "progress": "0",
            "created_at": str(time.time()),
        })
        q = Queue("eval", connection=self.redis)
        q.enqueue(run_eval_task, task_id, path, k, limit, job_timeout=3600)
        return task_id
    
    

    def get_eval_task(self, task_id: str) -> dict:
        data = self.redis.hgetall(self._get_task_key(task_id))
        if not data:
            return {"status": "not_found"}
        if "result" in data:
            data["result"] = json.loads(data["result"])
        return data

    
