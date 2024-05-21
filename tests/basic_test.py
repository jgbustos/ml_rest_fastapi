"""Basic pytest-based tests for ml_rest_api."""

from http import HTTPStatus
from json import loads
import requests
import pytest
from openapi_spec_validator import (
    validate,
    OpenAPIV2SpecValidator,
    OpenAPIV30SpecValidator,
    OpenAPIV31SpecValidator
)
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError


ROOT_URL = "http://localhost:8888/"
SWAGGER_JSON_PATH = "openapi.json"
DOCS_PATH = "docs"
REDOC_PATH = "redoc"
LIVENESS_PATH = "health/live"
READINESS_PATH = "health/ready"
MODEL_PREDICT_PATH = "model/predict"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
STRING_PARAM = "string_param"
INT_PARAM = "int_param"
FLOAT_PARAM = "float_param"
BOOL_PARAM = "bool_param"
DATETIME_PARAM = "datetime_param"
DATE_PARAM = "date_param"
GOOD_JSON_DICT = {
    STRING_PARAM: "foobar",
    INT_PARAM: 12345,
    FLOAT_PARAM: 123.45,
    BOOL_PARAM: True,
    DATETIME_PARAM: "2021-11-30T14:37:04.150Z",
    DATE_PARAM: "2021-10-26",
}
REQUEST_TIMEOUT = 0.1  # 100 milliseconds, life is too short

MOCK_PREDICTION = "mock_prediction"
NOT_A_DATETIME_MSG = "is not a 'date-time'"
NOT_A_DATE_MSG = "is not a 'date'"
VALUE_ERROR_MISSING = "missing"
VALUE_ERROR_DATE = "date_from_datetime_parsing"
VALUE_ERROR_DATETIME = "datetime_from_date_parsing"


def _get_request(url):
    response = requests.get(
        url,
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )
    return response


def _post_request(url, headers=None, json=None):
    response = requests.post(
        url,
        headers=headers,
        json=json,
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )
    return response


def _post_request_good_json_with_overrides(override_key=None, override_value=None):
    json_dict = GOOD_JSON_DICT
    if override_key is not None:
        if override_value is None:
            del json_dict[override_key]
        else:
            json_dict[override_key] = override_value
    response = _post_request(
        ROOT_URL + MODEL_PREDICT_PATH,
        headers=HEADERS,
        json=json_dict,
    )
    return response


def test_get_swagger_json_is_valid_openapi_v3_1():
    """Verify that /api/swagger.json file complies with OpenAPI v3.1"""
    response = _get_request(ROOT_URL + SWAGGER_JSON_PATH)
    spec_dict = loads(response.text)
    validate(spec_dict, cls=OpenAPIV31SpecValidator)
    assert response.status_code == HTTPStatus.OK


def test_get_swagger_json_is_not_valid_openapi_v3_0():
    """Verify that /api/swagger.json file does NOT comply with OpenAPI v3.0."""
    response = _get_request(ROOT_URL + SWAGGER_JSON_PATH)
    spec_dict = loads(response.text)
    with pytest.raises(OpenAPIValidationError):
        validate(spec_dict, cls=OpenAPIV30SpecValidator)
    assert response.status_code == HTTPStatus.OK


def test_get_swagger_json_is_not_valid_openapi_v2():
    """Verify that /api/swagger.json file does NOT comply with OpenAPI v2."""
    response = _get_request(ROOT_URL + SWAGGER_JSON_PATH)
    spec_dict = loads(response.text)
    with pytest.raises(OpenAPIValidationError):
        validate(spec_dict, cls=OpenAPIV2SpecValidator)
    assert response.status_code == HTTPStatus.OK


def test_get_docs_page_equals_200():
    """Verify that /docs succeeds with 200 and returns the Swagger UI."""
    response = _get_request(ROOT_URL + DOCS_PATH)
    assert response.status_code == HTTPStatus.OK
    assert "SwaggerUIBundle" in response.text


def test_get_redoc_page_equals_200():
    """Verify that /docs succeeds with 200 and returns the ReDoc UI."""
    response = _get_request(ROOT_URL + REDOC_PATH)
    assert response.status_code == HTTPStatus.OK
    assert "ReDoc requires Javascript to function" in response.text


def test_get_liveness_status_code_equals_200():
    """Verify that /health/live succeeds with 200 "Alive"."""
    response = _get_request(ROOT_URL + LIVENESS_PATH)
    assert response.status_code == HTTPStatus.OK
    assert "message" in response.text and "Live" in response.text


def test_get_readiness_status_code_equals_200():
    """Verify that /health/ready succeeds with 200 "Ready"."""
    response = _get_request(ROOT_URL + READINESS_PATH)
    assert response.status_code == HTTPStatus.OK
    assert "message" in response.text and "Ready" in response.text


def test_post_model_predict_no_headers_no_payload_status_code_equals_422():
    """Verify that calling /model/predict with no headers and no payload fails with 422."""
    response = _post_request(ROOT_URL + MODEL_PREDICT_PATH)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_post_model_predict_no_payload_status_code_equals_422():
    """Verify that calling /model/predict with no payload fails with 422."""
    response = _post_request(ROOT_URL + MODEL_PREDICT_PATH, headers=HEADERS)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_post_model_predict_empty_json_status_code_equals_422():
    """Verify that calling /model/predict with empty payload fails with 422 and flags all
    properties as required."""
    response = _post_request(ROOT_URL + MODEL_PREDICT_PATH, headers=HEADERS, json={})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert STRING_PARAM in response.text
    assert INT_PARAM in response.text
    assert FLOAT_PARAM in response.text
    assert BOOL_PARAM in response.text
    assert DATETIME_PARAM in response.text
    assert DATE_PARAM in response.text


def test_post_model_predict_good_json_status_code_equals_200():
    """Verify that calling /model/predict with good payload succeeds with 422 and returns
    MOCK_PREDICTION."""
    response = _post_request_good_json_with_overrides()
    assert response.status_code == HTTPStatus.OK
    assert MOCK_PREDICTION in response.text


def test_post_model_predict_datetime_param_missing_z_status_code_equals_200():
    """Verify that calling /model/predict with bad datetime_param (missing ending Z) works."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-30T14:37:04.15"
    )
    assert response.status_code == HTTPStatus.OK
    assert MOCK_PREDICTION in response.text


def test_post_model_predict_float_param_int_value_status_code_equals_200():
    """Verify that calling /model/predict with integer value as float_param works fine."""
    response = _post_request_good_json_with_overrides(FLOAT_PARAM, 10)
    assert response.status_code == HTTPStatus.OK
    assert MOCK_PREDICTION in response.text


def test_post_model_predict_bool_param_string_true_status_code_equals_200():
    """Verify that calling /model/predict with string "True" as bool_param works fine
    (yes, that's now pydantic rolls)."""
    response = _post_request_good_json_with_overrides(BOOL_PARAM, "True")
    assert response.status_code == HTTPStatus.OK
    assert MOCK_PREDICTION in response.text


def test_post_model_predict_string_param_float_status_code_equals_422():
    """Verify that calling /model/predict with float as string_param fails with 422."""
    response = _post_request_good_json_with_overrides(STRING_PARAM, 10.1)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "string_type" in response.text


def test_post_model_predict_int_param_float_status_code_equals_422():
    """Verify that calling /model/predict with float as int_param fails with 422."""
    response = _post_request_good_json_with_overrides(INT_PARAM, 10.1)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "int_from_float" in response.text


def test_post_model_predict_int_param_missing_status_code_equals_422():
    """Verify that calling /model/predict with no int_param fails with 422 and flags missing
    properties as required."""
    response = _post_request_good_json_with_overrides(INT_PARAM, None)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert INT_PARAM in response.text


def test_post_model_predict_string_param_missing_status_code_equals_422():
    """Verify that calling /model/predict with no string_param fails with 422 and flags missing
    properties as required."""
    response = _post_request_good_json_with_overrides(STRING_PARAM, None)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert STRING_PARAM in response.text


def test_post_model_predict_float_param_missing_status_code_equals_422():
    """Verify that calling /model/predict with no float_param fails with 422 and flags missing
    properties as required."""
    response = _post_request_good_json_with_overrides(FLOAT_PARAM, None)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert FLOAT_PARAM in response.text


def test_post_model_predict_bool_param_missing_status_code_equals_422():
    """Verify that calling /model/predict with no bool_param fails with 422 and flags missing
    properties as required."""
    response = _post_request_good_json_with_overrides(BOOL_PARAM, None)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert BOOL_PARAM in response.text


def test_post_model_predict_datetime_param_missing_status_code_equals_422():
    """Verify that calling /model/predict with no datetime_param fails with 422 and flags
    missing properties as required."""
    response = _post_request_good_json_with_overrides(DATETIME_PARAM, None)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_date_param_missing_status_code_equals_422():
    """Verify that calling /model/predict with no date_param fails with 422 and flags missing
    properties as required."""
    response = _post_request_good_json_with_overrides(DATE_PARAM, None)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_MISSING in response.text
    assert DATE_PARAM in response.text


def test_post_model_predict_int_param_string_status_code_equals_422():
    """Verify that calling /model/predict with string as int_param fails with 422 and flags
    wrong type."""
    response = _post_request_good_json_with_overrides(INT_PARAM, "a")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "int_parsing" in response.text
    assert INT_PARAM in response.text


def test_post_model_predict_float_param_string_status_code_equals_422():
    """Verify that calling /model/predict with string as float_param fails with 422 and flags
    wrong type."""
    response = _post_request_good_json_with_overrides(FLOAT_PARAM, "a")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "float_parsing" in response.text
    assert FLOAT_PARAM in response.text


def test_post_model_predict_bool_param_string_status_code_equals_422():
    """Verify that calling /model/predict with string as bool_param fails with 422 and flags
    wrong type."""
    response = _post_request_good_json_with_overrides(BOOL_PARAM, "foo")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "bool_parsing" in response.text
    assert BOOL_PARAM in response.text


def test_post_model_predict_datetime_param_bad_day_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (impossible day number) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-35T14:37:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_bad_month_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (impossible month number)
    fails with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-13-30T14:37:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_bad_year_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (impossible year number) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "999-11-30T14:37:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_missing_hyphen_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (missing date hyphen) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-1130T14:37:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_missing_t_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (missing T between date and
    time) fails with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-3014:37:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_missing_colon_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (missing time colon) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-30T1437:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_bad_hour_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (impossible hour number)
    fails with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-30T34:37:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_bad_minute_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (impossible minute number)
    fails with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-30T14:77:04.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_datetime_param_bad_second_status_code_equals_422():
    """Verify that calling /model/predict with bad datetime_param (impossible seconds number)
    fails with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(
        DATETIME_PARAM, "2021-11-30T14:37:64.15Z"
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATETIME in response.text
    assert DATETIME_PARAM in response.text


def test_post_model_predict_date_param_bad_year_status_code_equals_422():
    """Verify that calling /model/predict with bad date_param (impossible year number) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(DATE_PARAM, "999-10-26")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATE in response.text
    assert DATE_PARAM in response.text


def test_post_model_predict_date_param_bad_month_status_code_equals_422():
    """Verify that calling /model/predict with bad date_param (impossible month number) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(DATE_PARAM, "2021-18-26")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATE in response.text
    assert DATE_PARAM in response.text


def test_post_model_predict_date_param_bad_day_status_code_equals_422():
    """Verify that calling /model/predict with bad date_param (impossible day number) fails
    with 422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(DATE_PARAM, "2021-10-51")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATE in response.text
    assert DATE_PARAM in response.text


def test_post_model_predict_date_param_missing_hyphen_status_code_equals_422():
    """Verify that calling /model/predict with bad date_param (missing date hyphen) fails with
    422 and flags wrong type."""
    response = _post_request_good_json_with_overrides(DATE_PARAM, "2021-1026")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert VALUE_ERROR_DATE in response.text
    assert DATE_PARAM in response.text
