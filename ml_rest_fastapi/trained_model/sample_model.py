# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""

from time import sleep
from logging import Logger
from datetime import datetime, date, timezone
from os.path import normpath, join, dirname
from typing import Any, Iterable, Dict
import numpy as np
import pandas as pd
from ml_rest_fastapi.shared import get_logger
from ml_rest_fastapi.settings import get_value

# import joblib #NOSONAR

log: Logger = get_logger()


def full_path(filename: str) -> str:
    """
    Returns the full normalised path of a file in the same folder as this module.
    """
    return normpath(join(dirname(__file__), filename))


MODEL: Any = None


def init() -> None:
    """
    Loads the ML trained model (plus ancillary files) from file.
    """
    log.info("Load model from file: %s", full_path("model.pkl"))
    sleep(5)  # Fake delay to emulate a large model that takes a long time to load

    # deserialise the ML model (and possibly other objects such as feature_list,
    # feature_selector) from pickle file(s):
    #  global MODEL
    #  MODEL = joblib.load(full_path('model.pkl'))
    #  feature_list = joblib.load(full_path('feature_list.pkl'))
    #  feature_selector = joblib.load(full_path('feature_selector.pkl'))


def teardown() -> None:
    """
    Tears down the ML trained model
    """
    log.info("Tear down the model")
    sleep(5)  # Fake delay to emulate a large model that takes a long time to unload


def run(input_data: Iterable[Any]) -> Dict[str, Any]:
    """
    Makes a prediction using the trained ML model.
    """
    log.info("input_data: %s", input_data)
    data: pd.DataFrame = (
        input_data
        if isinstance(input_data, pd.DataFrame)
        else pd.DataFrame(input_data, index=[0])
    )

    # make the necessary transformations using pickled objects, e.g.
    #  data = pd.get_dummies(data)
    #  data = data.reindex(columns=feature_list, fill_value=0)
    #  data = feature_selector.transform(data)

    # then make (or mock) a prediction
    #  prediction = MODEL.predict(data)

    log.info(
        "transformed_data: %s", np.array_str(data.to_numpy()[0], max_line_width=10000)
    )
    prediction = "mock_prediction"
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]
    log.info("prediction: %s", prediction)

    ret: Dict[str, Any] = {}
    ret["prediction"] = prediction
    if get_value("EXPLAIN_PREDICTIONS"):
        ret["explanation"] = "mock_explanation"

    return ret


def sample() -> Dict[str, Any]:
    """
    Returns a sample input vector as a string-indexed dictionary of values.
    """
    return {
        "string_param": "foobar",
        "int_param": 42,
        "float_param": 2.71828,
        "bool_param": True,
        "datetime_param": datetime.now(tz=timezone.utc),
        "date_param": date.today(),
    }


if __name__ == "__main__":
    init()
    print(sample())
    print(run(sample()))
