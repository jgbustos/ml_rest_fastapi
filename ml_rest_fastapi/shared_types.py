"""This module declares shared types"""


class MLRestFastAPIException(Exception):
    """Base ML Rest FastAPI Exception"""


class MLRestFastAPINotReadyException(MLRestFastAPIException):
    """Base ML Rest FastAPI NOT READY Exception"""
