FROM python:3.14.0a1-slim

LABEL maintainer="j.garciadebustos@godeltech.com"

# OpenMP library needed by LightGBM, XGBoost, etc
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && apt clean

COPY requirements.txt /app/

RUN pip install --upgrade pip \
 && pip install --upgrade --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8888
ENV PYTHONPATH "${PYTHONPATH}:/app/ml_rest_fastapi"

CMD ["/usr/local/bin/gunicorn", "--config", "gunicorn.conf.py"]
