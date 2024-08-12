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
    actual = DataClusters(np.array([1, 2, 3]), np.array([3, 2, 1]), 0)
    expected = DataClusters(np.array([1, 2, 3]), np.array([3, 2, 1]), 0)
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

    # In the set data test, we test for these cases.
    # (1) x and y are non-empty array values, and res is positive (the most generic case)
    # (2) x and y are non-empty array values, and res is 0 (will produce a msg that makes trivial clustering)
    # (3) x and y are non-empty array values, and res is negative (will produce a ValueError,
    # msg = please enter a non-negative res value)
    # (4, 5) One of x and y is empty array, and res is positive
    # (produce ValueError & msg "Sequences x and y must have the same length.", something like that)
    # (6) Both x and y are empty array, and res is zero.


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            # case (1)
            {
                "input_x": np.array([1, 2, 3]),
                "input_y": np.array([3, 2, 1]),
                "input_res": 4,
            },
            DataClusters(np.array([1, 2, 3]), np.array([3, 2, 1]), 4),
        ),
        (
            # case (6)
            {
                "input_x": np.array([]),
                "input_y": np.array([]),
                "input_res": 0,
            },
            DataClusters(np.array([]), np.array([]), 0),
        ),
    ],
)
def test_set_data(inputs, expected):
    actual = DataClusters(x=inputs["input_x"], y=inputs["input_y"], res=inputs["input_res"])
    assert actual == expected


@pytest.mark.parametrize(
    "inputs, msg",
    [
        (
            # case (4)
            {
                "input_x": np.array([]),
                "input_y": np.array([3, 2]),
                "input_res": 4,
            },
            "Sequences x and y must have the same length.",
        ),
        (
            # case (5)
            {
                "input_x": np.array([1, 2]),
                "input_y": np.array([]),
                "input_res": 4,
            },
            "Sequences x and y must have the same length.",
        ),
        (
            # case (3)
            {
                "input_x": np.array([1]),
                "input_y": np.array([3]),
                "input_res": -1,
            },
            "Resolution res must be non-negative.",
        ),
        (
            # case (2)
            {
                "input_x": np.array([1, 2, 3]),
                "input_y": np.array([3, 2, 1]),
                "input_res": 0,
            },
            "Make trivial clustering, please make positive resolution.",
        ),
    ],
)
def test_set_data_order_bad(inputs, msg):
    with pytest.raises(ValueError, match=msg):
        DataClusters(x=inputs["input_x"], y=inputs["input_y"], res=inputs["input_res"])
