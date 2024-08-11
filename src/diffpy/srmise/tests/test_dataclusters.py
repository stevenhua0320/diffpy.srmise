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


# In the set data test, we test for these cases.
# (1) x and y are non-empty array values, and res is positive (the most generic case)
# (2) x and y are non-empty array values, and res is 0 (will produce a ValueError)
# (3) x and y are non-empty array values, and res is negative (will produce a ValueError,
# but give different msg than 2)
# (4) x and y are empty array, and res is positive (produce ValueError & msg "please give input to x/y array",
# something like that)
# (5) Same as 4, except res is 0 (Initialized state)
# (6) Same as 4, except res is negative (ValueError)
# (7) Same as 1/2/3, except, x & y have different length (ValueError & msg)


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
