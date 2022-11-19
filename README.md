# ml_rest_fastapi

[![Sonarcloud Status](https://sonarcloud.io/api/project_badges/measure?project=jgbustos_ml_rest_fastapi&metric=alert_status)](https://sonarcloud.io/dashboard?id=jgbustos_ml_rest_fastapi)
[![Known Vulnerabilities](https://snyk.io/test/github/jgbustos/ml_rest_fastapi/badge.svg)](https://app.snyk.io/org/jgbustos/projects)

A RESTful API to return predictions from a trained ML model, built with Python 3 and FastAPI

## Development set-up instructions

First, open a command line interface and clone the GitHub repo in your workspace

```Powershell
PS > cd $WORKSPACE_PATH$
PS > git clone https://github.com/jgbustos/ml_rest_fastapi
PS > cd ml_rest_fastapi
```

Create and activate a Python virtual environment, then install the required Python packages using pip

```Powershell
PS > virtualenv venv
PS > venv\scripts\activate.ps1
(venv) PS > pip install -r requirements.txt
```

Once dependencies are installed, set up the project for development

```Powershell
(venv) PS > pip install -e .
```

Finally, run the project:

```Powershell
(venv) PS > uvicorn ml_rest_fastapi.app:app --host 0.0.0.0 --port 8888 --reload
```

Open the URL <http://localhost:8888/docs/> with your browser and see the sample OpenAPI documentation

## Interfaces exposed

OpenAPI JSON available from URL <http://localhost:8888/openapi.json>

### Health

These two methods are meant to be used as the liveness and readiness probes in a Kubernetes deployment:

* GET <http://localhost:8888/health/live> returns 200/"Live" if the service is up and running
* GET <http://localhost:8888/health/ready> returns 200/"Ready" or 503/"Not Ready" depending on whether the ML model has been correctly initialised or not

### Model

* POST <http://localhost:8888/model/predict> will return a prediction using the ML model. The data_point structure shows the JSON argument that must be supplied, and example values for each of the fields. The service will validate that all the mandatory values are passed. Return values are:
  * 200/Predicted value based on JSON input
  * 422/"Validation Error" if any mandatory parameter is missing or if any wrong data type (e.g. str, int, bool, datetime...) is supplied
  * 500/"Internal Server Error" as catch-all exception handler
  * 503/"Not Ready" if model is not initialised

## Config settings

Configuration parameters are contained in the file **ml_rest_fastapi/settings.py**, but they can also be overriden by setting env vars:

```python
settings: Dict = {
    # Trained ML/AI model settings
    "TRAINED_MODEL_MODULE_NAME": "sample_model",
    # Module settings
    "DEBUG": True,
    "MULTITHREADED_INIT": True,
}
```

| Parameter | Values | Details |
| --- | --- | --- |
| TRAINED_MODEL_MODULE_NAME | e.g.: sample_model | Name of the Python module that initialises the ML model and returns predictions (see [section below](#setting-up-the-model)) |
| DEBUG | False/True | When Debug = True, Uvicorn will run in reload mode, and FastAPI will return a colourful traceback instead of a sober 500/"Internal Server Error" |
| MULTITHREADED_INIT | False/True | Multi-threaded initialisation of the trained model |

## Setting up the model

The trained ML model is meant to be initialised and invoked to make predictions in the context of a Python unit saved inside the directory **ml_rest_fastapi/trained_model**. The structure of this Python module is explained in [this document](ml_rest_fastapi/trained_model/module_structure.md)
