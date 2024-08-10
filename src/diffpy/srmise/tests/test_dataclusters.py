import numpy as np
import pytest

from diffpy.srmise.dataclusters import DataClusters


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            {
                "input_x": np.array([1, 2, 3]),
                "input_y": np.array([3, 2, 1]),
                "input_res": 4,
            },
            {
                "x": np.array([]),
                "y": np.array([]),
                "res": 0,
            },
        ),
    ],
)
def test_clear(inputs, expected):
    # Initialize DataClusters with input parameters
    actual = DataClusters(x=inputs["input_x"], y=inputs["input_y"], res=inputs["input_res"])
    expected = DataClusters(x=expected["x"], y=expected["y"], res=expected["res"])
    # Perform the clear operation
    actual.clear()
    assert actual == expected
