"""This module implements the health methods"""

from fastapi import APIRouter, HTTPException
from ml_rest_fastapi.trained_model.wrapper import trained_model_wrapper


health_route = APIRouter()


@health_route.get(
    "/live",
    summary="Returns liveness status",
    operation_id="liveness_get",
    responses={
        200: {
            "description": "Success",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Live",
                }
            },
        },
    },
)
def liveness():
    """
    Returns liveness status.
    """
    return "Live"


@health_route.get(
    "/ready",
    summary="Returns readiness status",
    operation_id="readiness_get",
    responses={
        200: {
            "description": "Success",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Ready",
                }
            },
        },
        503: {
            "description": "Server Not Ready",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Not Ready",
                }
            },
        },
    },
)
def readiness():
    """
    Returns readiness status.
    """
    if not trained_model_wrapper.initialised:
        raise HTTPException(503, detail="Not Ready")
    return "Ready"
