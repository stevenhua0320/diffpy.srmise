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
                "data_order": np.array([], dtype=np.int32),
                "clusters": np.array([[]], dtype=np.int32),
                "res": 0,
                "current_idx": 0,
                "lastcluster_idx": None,
                "lastpoint_idx": None,
                "status": 0,
            },
        ),
    ],
)
def test_clear(inputs, expected):
    # Initialize DataClusters with input parameters
    c1 = DataClusters(x=inputs["input_x"], y=inputs["input_y"], res=inputs["input_res"])

    # Perform the clear operation
    c1.clear()

    # Assert each expected attribute against its actual value after clearing
    for attr, expected_value in expected.items():
        assert (
            np.array_equal(getattr(c1, attr), expected_value)
            if isinstance(expected_value, np.ndarray)
            else getattr(c1, attr) == expected_value
        )
