from copy import copy

import numpy as np

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
