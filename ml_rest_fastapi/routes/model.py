"""This module implements the model inference methods"""

from fastapi import APIRouter

model_route = APIRouter()


@model_route.post(
    "/predict",
    summary="Returns a prediction using the trained ML model",
    operation_id="predict_post",
    responses={
        200: {
            "description": "Success",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Success",
                }
            },
        },
        400: {
            "description": "Input Validation Error",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Input Validation Error",
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Internal Server Error",
                }
            },
        },
        503: {
            "description": "Server Not Ready",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Server Not Ready",
                }
            },
        },
    },
)
def model_predict():
    """Returns a prediction using the trained ML model."""
    return {"prediction": "mock_prediction"}
