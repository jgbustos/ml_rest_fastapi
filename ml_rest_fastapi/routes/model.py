"""This module implements the model inference methods"""

from typing import Dict, TYPE_CHECKING
from fastapi import APIRouter, status
from pydantic import BaseModel, create_model

from ml_rest_fastapi.settings import get_value
from ml_rest_fastapi.shared_types import (
    Message,
    MLRestFastAPINotReadyException,
)
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


responses_dict: Dict = {
    status.HTTP_200_OK: {
        "model": Dict,
        "content": {
            "application/json": {
                "example": {"prediction": "example"},
            }
        },
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {},
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": Message,
        "content": {
            "application/json": {
                "example": Message("Not Ready").to_json(),
            }
        },
    },
}

if get_value("DEBUG"):
    responses_dict[status.HTTP_500_INTERNAL_SERVER_ERROR]["content"] = {
        "text/plain": {
            "example": """Traceback (most recent call last):
  File "xxxxx.py", line 1234, in __call__
    return await foo
  File "yyyyy.py", line 5678, in __call__
    return bar[0]
IndexError: list index out of range"""
        }
    }
else:
    responses_dict[status.HTTP_500_INTERNAL_SERVER_ERROR]["model"] = Message
    responses_dict[status.HTTP_500_INTERNAL_SERVER_ERROR]["content"] = {
        "application/json": {
            "example": Message("Internal Server Error").to_json(),
        }
    }


@model_route.post(
    "/predict",
    summary="Returns a prediction using the trained ML model",
    operation_id="predict_post",
    responses=responses_dict,
)
def model_predict(input_vector: InputVector) -> Dict:
    """
    Returns a prediction using the trained ML model.
    """
    if not trained_model_wrapper.initialised:
        raise MLRestFastAPINotReadyException()
    data = dict(input_vector)
    prediction = trained_model_wrapper.run(data)
    return prediction
