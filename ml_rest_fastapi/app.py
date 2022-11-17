"""This module is the RESTful service entry point."""

import uvicorn
from fastapi import FastAPI

from ml_rest_fastapi.routes.health import health_route
from ml_rest_fastapi.routes.model import model_route
from ml_rest_fastapi.settings import get_value
from ml_rest_fastapi.trained_model.wrapper import trained_model_wrapper

tags_metadata = [
    {
        "name": "health",
        "description": "Basic health check methods",
    },
    {
        "name": "model",
        "description": "Methods to return a prediction from the trained ML model",
    },
]


app = FastAPI(
    title="Machine Learning REST FastAPI",
    description="A RESTful API to return predictions from a trained ML model, \
        built with Python 3 and FastAPI",
    version="0.1.0",
    openapi_tags=tags_metadata,
)


app.include_router(health_route, prefix="/health", tags=["health"])
app.include_router(model_route, prefix="/model", tags=["model"])


@app.on_event("startup")
def startup_event():
    """
    FastAPI startup event, used to initialise the trained ML model wrapper.
    """
    if get_value("MULTITHREADED_INIT"):
        trained_model_wrapper.multithreaded_init()
    else:
        trained_model_wrapper.init()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8888, reload=True)