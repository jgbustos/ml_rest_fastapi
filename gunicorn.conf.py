import multiprocessing

wsgi_app = "ml_rest_fastapi.app:app"
bind = "0.0.0.0:8888"
workers = multiprocessing.cpu_count() * 2 - 1
worker_class = "uvicorn.workers.UvicornWorker"
accesslog = "-"
errorlog = "-"
