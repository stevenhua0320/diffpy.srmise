from copy import copy

import numpy as np
import pytest

from diffpy.srmise.dataclusters import DataClusters


def test_clear():
    # Initialize DataClusters with input parameters
    actual = DataClusters(x=np.array([1, 2, 3]), y=np.array([3, 2, 1]), res=4)
    expected = DataClusters(x=np.array([]), y=np.array([]), res=0)
    # Perform the clear operation
    actual.clear()
    assert actual == expected


def test___eq__():
    actual = DataClusters(np.array([1, 2, 3]), np.array([3, 2, 1]), 1)
    expected = DataClusters(np.array([1, 2, 3]), np.array([3, 2, 1]), 1)
    assert expected == actual
    attributes = vars(actual)
    for attr_key, attr_val in attributes.items():
        reset = copy(attr_val)
        assert expected == actual
        if attr_val is not None:
            attributes.update({attr_key: attr_val + 1})
        else:
            attributes.update({attr_key: 1})
        try:
            assert not expected == actual
        except AssertionError:
            print(f"not-equal test failed on {attr_key}")
            assert not expected == actual
        attributes.update({attr_key: reset})


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            {
                "x": np.array([1, 2, 3]),
                "y": np.array([3, 2, 1]),
                "res": 4,
            },
            DataClusters(x=np.array([1, 2, 3]), y=np.array([3, 2, 1]), res=4),
        ),
    ],
)
def test_set_data(inputs, expected):
    actual = DataClusters(x=np.array([]), y=np.array([]), res=0)
    actual.setdata(x=inputs["x"], y=inputs["y"], res=inputs["res"])
    assert expected == actual


@pytest.mark.parametrize(
    "inputs, msg",
    [
        (
            {
                "x": np.array([1]),
                "y": np.array([3, 2]),
                "res": 4,
            },
            "Sequences x and y must have the same length.",
        ),
        (
            {
                "x": np.array([1]),
                "y": np.array([3]),
                "res": -1,
            },
            "Value of resolution parameter is less than zero.  Please rerun specifying a non-negative res",
        ),
    ],
)
def test_set_data_order_bad(inputs, msg):
    with pytest.raises(ValueError, match=msg):
        DataClusters(x=inputs["x"], y=inputs["y"], res=inputs["res"])
