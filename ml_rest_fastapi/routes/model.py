"""This module implements the model inference methods"""

from typing import Dict, TYPE_CHECKING
from fastapi import APIRouter
from pydantic import BaseModel, create_model

from ml_rest_fastapi.shared_types import MLRestFastAPINotReadyException
from ml_rest_fastapi.trained_model.wrapper import trained_model_wrapper

# mypy really doesn't like dynamically-generated types
# See https://github.com/pydantic/pydantic/issues/615
if TYPE_CHECKING:
    # This is an alias, so mypy is happy with it
    # Therefore, this code runs during type checking
    InputVector = BaseModel
else:
    # Whereas this is a variable, so mypy complains if you use it as an annotation
    # However, this is perfectly fine (and needed!) in run-time
    InputVector = create_model("InputVector", **trained_model_wrapper.sample())

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
            "description": "Error: Service Unavailable",
            "model": str,
            "content": {
                "application/json": {
                    "example": "Not Ready",
                }
            },
        },
    },
)
def model_predict(input_vector: InputVector) -> Dict:
    """
    Returns a prediction using the trained ML model.
    """
    if not trained_model_wrapper.initialised:
        raise MLRestFastAPINotReadyException()
    data = dict(input_vector)
    ret = trained_model_wrapper.run(data)
    return ret
