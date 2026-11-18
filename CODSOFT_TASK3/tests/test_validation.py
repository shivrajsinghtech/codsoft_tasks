import math

import pytest

from src.predict import InvalidInputError, validate_customer


def test_valid_input_passes(valid_customer):
    out = validate_customer(valid_customer)
    assert out["Geography"] == "France"
    assert isinstance(out["Age"], int)


def test_case_insensitive_categories_and_numeric_strings(valid_customer):
    valid_customer.update(Geography="germany", Gender=" male ", Age="40")
    out = validate_customer(valid_customer)
    assert out["Geography"] == "Germany" and out["Gender"] == "Male" and out["Age"] == 40


@pytest.mark.parametrize("field,value", [
    ("CreditScore", 100), ("CreditScore", 1000), ("Age", 10), ("Age", 150),
    ("Tenure", -1), ("Tenure", 11), ("Balance", -5), ("NumOfProducts", 0),
    ("NumOfProducts", 5), ("HasCrCard", 2), ("IsActiveMember", -1),
    ("EstimatedSalary", -1), ("Geography", "Italy"), ("Gender", "Other"),
    ("Age", "abc"), ("Age", 30.5), ("Balance", math.nan), ("Balance", math.inf),
])
def test_invalid_values_rejected(valid_customer, field, value):
    valid_customer[field] = value
    with pytest.raises(InvalidInputError) as exc:
        validate_customer(valid_customer)
    assert any(field in e for e in exc.value.errors)


@pytest.mark.parametrize("field", ["Age", "Geography", "Balance", "IsActiveMember"])
def test_missing_field_rejected(valid_customer, field):
    del valid_customer[field]
    with pytest.raises(InvalidInputError, match=f"{field} is required"):
        validate_customer(valid_customer)


def test_none_and_blank_rejected(valid_customer):
    valid_customer["Gender"] = None
    valid_customer["Geography"] = "  "
    with pytest.raises(InvalidInputError) as exc:
        validate_customer(valid_customer)
    assert len(exc.value.errors) == 2


def test_non_mapping_rejected():
    with pytest.raises(InvalidInputError):
        validate_customer(["not", "a", "dict"])
