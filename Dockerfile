FROM python:3.10-slim

LABEL maintainer="j.garciadebustos@godeltech.com"

COPY requirements.txt /app/

RUN pip install --upgrade pip \
 && pip install --upgrade --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888
ENV PYTHONPATH "${PYTHONPATH}:/app/ml_rest_fastapi"

CMD ["/usr/local/bin/uvicorn", "ml_rest_fastapi.app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8888"]
