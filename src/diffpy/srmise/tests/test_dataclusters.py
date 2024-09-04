from copy import copy

import numpy as np
import pytest

from diffpy.srmise.dataclusters import DataClusters


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
            {
                "x": np.array([1, 2, 3]),
                "y": np.array([3, 2, 1]),
                "res": 4,
                "data_order": [2, 1, 0],
                "clusters": np.array([[0, 0]]),
                "current_idx": 2,
                "lastpoint_idx": 0,
                "INIT": 0,
                "READY": 1,
                "CLUSTERING": 2,
                "DONE": 3,
                "lastcluster_idx": None,
                "status": 1,
            },
        ),
    ],
)
def test_DataClusters_constructor(inputs, expected):
    actual = DataClusters(x=inputs["x"], y=inputs["y"], res=inputs["res"])
    actual_attributes = vars(actual)
    for attr_key, actual_attr_val in actual_attributes.items():
        if isinstance(actual_attr_val, np.ndarray):
            assert np.array_equal(actual_attr_val, expected[attr_key])
        else:
            assert actual_attr_val == expected[attr_key]


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
