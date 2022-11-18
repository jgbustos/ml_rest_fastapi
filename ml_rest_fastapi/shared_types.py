"""This module declares shared types"""

from dataclasses import dataclass, asdict


@dataclass
class Serialisable:
    """Parent serialisable dataclass"""

    def to_json(self):
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
