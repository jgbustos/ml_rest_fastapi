# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the server (Windows, required before running tests)
```bash
uvicorn ml_rest_fastapi.app:app --host 0.0.0.0 --port 8888 --reload
```

### Run the server (Linux)
```bash
gunicorn -c gunicorn.conf.py
```

### Run all tests (server must be running first)
```bash
pytest tests/
```

### Run a single test
```bash
pytest tests/basic_test.py::test_get_liveness_status_code_equals_200
```

### Lint and type-check (from repo root, with venv active)
```bash
bash make.sh
# or on Windows PowerShell:
# ./make.ps1
```
This runs `black` (formatting), `pylint` (linting), and `mypy` (type checking) in sequence.

### Individual tools
```bash
black ./ml_rest_fastapi ./tests
pylint --recursive=y ./ml_rest_fastapi ./tests
mypy --pretty --config-file=mypy.ini ./ml_rest_fastapi
```

### Dependencies
- `requirements.txt` — runtime dependencies (install with `pip install -r requirements.txt`)
- `requirements-dev.txt` — dev tooling: black, pylint, mypy
- `tests/requirements.txt` — test dependencies: pytest, requests, openapi-spec-validator

## Architecture

The core design pattern is a **plugin-style model wrapper**: `TrainedModelWrapper` (`trained_model/wrapper.py`) dynamically imports a Python module from `trained_model/` at startup and binds four callables from it: `init()`, `teardown()`, `run(data)`, and `sample()`. The active module is chosen by the `TRAINED_MODEL_MODULE_NAME` setting (env var overrides `settings.py`).

**Critical startup sequence:**
1. At module import time, `wrapper.py` instantiates `trained_model_wrapper` and calls `load_default_module()` — this imports the model module.
2. `routes/model.py` then calls `trained_model_wrapper.sample()` at import time to dynamically build the `InputVector` Pydantic model via `pydantic.create_model`. This means `sample()` must be callable before the app starts.
3. On server startup, `app.py`'s lifespan context manager calls `trained_model_wrapper.setup()`, which runs `init()` (optionally in a background thread if `MULTITHREADED_INIT=True`).
4. `health/ready` returns 503 until `init()` completes.

**Adding a new model module:** Create `ml_rest_fastapi/trained_model/<name>.py` implementing `init()`, `teardown()`, `run(data: Iterable) -> Iterable`, and `sample() -> Dict` with mypy type hints. Set `TRAINED_MODEL_MODULE_NAME=<name>`. See `sample_model.py` for a template and `adult_census_income.py` for a real LightGBM example.

**Logging:** All logging goes through [loguru](https://github.com/Delgan/loguru). `app.py` installs an `InterceptHandler` at startup that redirects the stdlib root logger and all uvicorn/gunicorn named loggers into loguru. Model modules import the logger directly: `from loguru import logger as log`.

**Settings** (`settings.py`): All settings fall back from env var → `settings` dict. Key settings: `TRAINED_MODEL_MODULE_NAME`, `EXPLAIN_PREDICTIONS`, `DEBUG`, `MULTITHREADED_INIT`.

**Tests** in `tests/basic_test.py` are integration tests that hit a live server at `http://localhost:8888/`. They test the `sample_model` module by default; the payload shape matches what `sample()` returns.
