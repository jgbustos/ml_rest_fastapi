"""This module implements the health methods"""

from fastapi import APIRouter, status

from ml_rest_fastapi.shared import (
    Message,
    MLRestFastAPINotReadyException,
    make_openapi_response,
)
from ml_rest_fastapi.trained_model.wrapper import trained_model_wrapper

health_route = APIRouter()


@health_route.get(
    "/live",
    summary="Returns liveness status",
    operation_id="liveness_get",
    responses={
        status.HTTP_200_OK: make_openapi_response(
            model_type=Message, example=Message("Live").to_json()
        ),
    },
)
def liveness() -> Message:
    """
    Returns liveness status.
    """
    return Message("Live")


@health_route.get(
    "/ready",
    summary="Returns readiness status",
    operation_id="readiness_get",
    responses={
        status.HTTP_200_OK: make_openapi_response(
            model_type=Message, example=Message("Ready").to_json()
        ),
        status.HTTP_503_SERVICE_UNAVAILABLE: make_openapi_response(
            model_type=Message, example=Message("Not Ready").to_json()
        ),
    },
)
def readiness() -> Message:
    """
    Returns readiness status.
    """
    if not trained_model_wrapper.initialised:
        raise MLRestFastAPINotReadyException()
    return Message("Ready")
