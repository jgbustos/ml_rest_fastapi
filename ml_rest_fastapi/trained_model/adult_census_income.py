# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""
from logging import Logger
from os.path import normpath, join, dirname
from typing import Any, Iterable, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import eli5
from ml_rest_fastapi.shared import get_logger
from ml_rest_fastapi.settings import get_value


log: Logger = get_logger()


def full_path(filename: str) -> str:
    """
    Returns the full normalised path of a file in the same folder as this module.
    """
    return normpath(join(dirname(__file__), filename))


# Pickled model
MODEL_PICKLE_FILE: str = "LightGBM_80.pkl"
# Pickled list of columns after pandas.get_dummies()
COLUMNS_PICKLE_FILE: str = "columns.pkl"

MODEL: Any = None
COLUMNS: Any = None


def init() -> None:
    """
    Loads the ML trained model (plus ancillary files) from file.
    """
    log.info("Load model from file: %s", full_path(MODEL_PICKLE_FILE))
    global MODEL  # pylint: disable=global-statement
    global COLUMNS  # pylint: disable=global-statement
    MODEL = joblib.load(full_path(MODEL_PICKLE_FILE))
    COLUMNS = joblib.load(full_path(COLUMNS_PICKLE_FILE))


def run(input_data: Iterable) -> Dict:
    """
    Makes a prediction using the trained ML model.
    """
    log.info("input_data: %s", input_data)
    data = pd.DataFrame(input_data, index=[0])
    data = pd.get_dummies(data)
    data = data.reindex(columns=COLUMNS, fill_value=0)
    log.info("transformed_data: %s", np.array_str(data.values[0], max_line_width=10000))

    ret = {}

    prediction = MODEL.predict(data)
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()[0]
    log.info("prediction: %s", prediction)
    ret["prediction"] = prediction

    if get_value("EXPLAIN_PREDICTIONS"):
        estimator = MODEL
        transformed_data = data
        feature_names = COLUMNS

        # MODEL might be a Pipeline with an estimator at the end, possibly even a feature selector
        if isinstance(estimator, Pipeline):
            if len(estimator.steps) > 1:
                transformer = Pipeline(estimator.steps[:-1])
                transformed_data = pd.DataFrame(transformer.transform(data), index=[0])
                for step in transformer.steps:
                    if hasattr(step[1], "get_support"):  # feature selector
                        feature_names = data.columns[step[1].get_support()].tolist()
                        break
            estimator = estimator.steps[-1][1]
        ret["explanation"] = eli5.format_as_dict(
            eli5.explain_prediction(
                estimator, transformed_data, feature_names=feature_names, top=None
            )
        )
    return ret


def sample() -> Dict[str, Any]:
    """
    Returns a sample input vector as a string-indexed dictionary of values.
    """
    return {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 100000,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United States",
    }


if __name__ == "__main__":
    init()
    print(sample())
    print(run(sample()))
