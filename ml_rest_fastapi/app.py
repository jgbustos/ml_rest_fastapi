"""This module is the RESTful service entry point."""

import os
import platform
from subprocess import run
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
from starlette.applications import Starlette
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ml_rest_fastapi.settings import get_value
from ml_rest_fastapi.shared import (
    Message,
    MLRestFastAPINotReadyException,
)
from ml_rest_fastapi.routes.health import health_route
from ml_rest_fastapi.routes.model import model_route
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


@asynccontextmanager
async def lifespan(_fastapi: Starlette) -> AsyncGenerator:
    """
    Lifespan context manager, used to initialise and teardown the trained ML model wrapper
    """
    if get_value("MULTITHREADED_INIT"):
        trained_model_wrapper.multithreaded_init()
    else:
        trained_model_wrapper.init()
    yield
    trained_model_wrapper.teardown()


app = FastAPI(
    title="Machine Learning REST FastAPI",
    description="A RESTful API to return predictions from a trained ML model, \
        built with Python 3 and FastAPI",
    version="0.1.0",
    openapi_tags=tags_metadata,
    debug=get_value("DEBUG"),
    lifespan=lifespan,
)


app.include_router(health_route, prefix="/health", tags=["health"])
app.include_router(model_route, prefix="/model", tags=["model"])


@app.exception_handler(MLRestFastAPINotReadyException)
def not_ready_exception_handler(
    request: Request,  # pylint: disable=unused-argument
    exc: MLRestFastAPINotReadyException,  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    "Not Ready" exception handler that returns HTTP 503 error.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=Message("Not Ready").to_json(),
    )


if __name__ == "__main__":
    if platform.uname().system.lower() == "linux":
        run(
            [
                "gunicorn",
                "-c",
                os.path.normpath(os.path.dirname(__file__) + "/../gunicorn.conf.py"),
            ],
            check=True,
        )
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8888, reload=get_value("DEBUG"))
