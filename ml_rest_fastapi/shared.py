"""This module declares shared stuff"""

from typing import Any, Type, Optional, Dict, Union
from logging import Logger, getLogger
from dataclasses import dataclass, asdict
from ml_rest_fastapi.settings import get_value


@dataclass
class Serialisable:
    """Parent serialisable dataclass"""

    def to_json(self) -> Dict[str, Any]:
        """Return as dict so it can be JSON serialisable"""
        return asdict(self)


@dataclass
class Message(Serialisable):
    """Dataclass for simple responses"""

    message: str


class MLRestFastAPIException(Exception):
    """Base ML Rest FastAPI Exception"""


class MLRestFastAPINotReadyException(MLRestFastAPIException):
    """Base ML Rest FastAPI NOT READY Exception"""


def make_openapi_response(
    model_type: Optional[Type[Any]] = None,
    mime_type: str = "application/json",
    example: Union[str, Dict[str, Any]] = "",
) -> Dict[str, Any]:
    """
    Returns a declaration of an OpenAPI extended response.
    """
    ret: Dict[str, Any] = {}
    if model_type:
        ret["model"] = model_type
    ret["content"] = {
        mime_type: {
            "example": example,
        },
    }
    return ret


def get_logger() -> Logger:
    """
    Return the uvicorn or gunicorn logger instance, based on how we're running
    """
    if "gunicorn" in str(get_value("SERVER_SOFTWARE")):
        log = getLogger("gunicorn.error")
    else:
        log = getLogger("uvicorn.error")
    return log
