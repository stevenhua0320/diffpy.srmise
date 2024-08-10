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


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            {
                "input_x": np.array([1, 2, 3]),
                "input_y": np.array([3, 2, 1]),
                "input_res": 4,
            },
            DataClusters(np.array([1, 2, 3]), np.array([3, 2, 1]), 4),
        ),
        (
            {
                "input_x": np.array([1]),
                "input_y": np.array([3]),
                "input_res": 1,
            },
            DataClusters(np.array([1]), np.array([3]), 1),
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
            {
                "input_x": np.array([1, 2, 3]),
                "input_y": np.array([3, 2]),
                "input_res": 4,
            },
            "Sequences x and y must have the same length.",
        ),
        (
            {
                "input_x": np.array([1]),
                "input_y": np.array([3]),
                "input_res": -1,
            },
            "Resolution res must be non-negative.",
        ),
        (
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
