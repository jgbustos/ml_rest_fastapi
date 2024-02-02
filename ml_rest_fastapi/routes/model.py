"""This module implements the model inference methods"""

from typing import Any, Dict, Union, TYPE_CHECKING
from fastapi import APIRouter, status
from pydantic import BaseModel, create_model

from ml_rest_fastapi.settings import get_value
from ml_rest_fastapi.shared import (
    Message,
    MLRestFastAPINotReadyException,
    make_openapi_response,
)
from ml_rest_fastapi.trained_model.wrapper import trained_model_wrapper


def dict_value_types(arg: Dict) -> Dict:
    """
    Takes a Dict like {"foo": "bar", "xx": 12} and returns {"foo": (str, ...), "xx": (int, ...)}
    because pydantic.create_model needs these (type, ellipsis) tuples to make fields required
    """
    return {key: (type(value), ...) for key, value in arg.items()}


# mypy really doesn't like dynamically-generated types
# See https://github.com/pydantic/pydantic/issues/615
if TYPE_CHECKING:
    # This is an alias, so mypy is happy with it
    # Therefore, this code runs during type checking
    InputVector = BaseModel
else:
    # Whereas this is a variable, so mypy complains if you use it as an annotation
    # However, this is perfectly fine (and needed!) in run-time
    InputVector = create_model(
        "InputVector", **dict_value_types(trained_model_wrapper.sample())
    )

model_route = APIRouter()


EXAMPLE_TRACEBACK: str = """Traceback (most recent call last):
  File "xxxxx.py", line 1234, in __call__
    return await foo
  File "yyyyy.py", line 5678, in __call__
    return bar[0]
IndexError: list index out of range"""


responses_dict: Dict[Union[int, str], Dict[str, Any]] = {
    status.HTTP_200_OK: make_openapi_response(
        model_type=Dict, example={"prediction": "example"}
    ),
    status.HTTP_500_INTERNAL_SERVER_ERROR: (
        make_openapi_response(mime_type="text/plain", example=EXAMPLE_TRACEBACK)
        if get_value("DEBUG")
        else make_openapi_response(
            model_type=Message, example=Message("Internal Server Error").to_json()
        )
    ),
    status.HTTP_503_SERVICE_UNAVAILABLE: make_openapi_response(
        model_type=Message, example=Message("Not Ready").to_json()
    ),
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
