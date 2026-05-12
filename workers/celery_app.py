from dotenv import load_dotenv
load_dotenv()  # must run before any API client reads os.environ

from celery import Celery
from config.settings import config

celery_app = Celery(
    "resume_matcher",
    broker=config.redis_url,
    backend=config.redis_url,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # solo pool avoids fork() crashes on macOS with PyTorch/sentence-transformers.
    # Switch to "prefork" in production on Linux.
    worker_pool="solo",
)
