import numpy as np
import pytest

from diffpy.srmise.dataclusters import DataClusters


@pytest.mark.parametrize(
    "input_x, input_y, input_res, expected_x, expected_y, expected_data_order, expected_clusters, expected_res, "
    "expected_current_index, expected_last_cluster_idx, expected_last_pt_idx, expected_status",
    [
        (
            np.array([1, 2, 3]),
            np.array([3, 2, 1]),
            4,
            np.array([]),
            np.array([]),
            np.array([], dtype=np.int32),
            np.array([[]], dtype=np.int32),
            0,
            0,
            None,
            None,
            0,
        )
    ],
)
def test_clear(
    input_x,
    input_y,
    input_res,
    expected_x,
    expected_y,
    expected_data_order,
    expected_clusters,
    expected_res,
    expected_current_index,
    expected_last_cluster_idx,
    expected_last_pt_idx,
    expected_status,
):
    c1 = DataClusters(x=input_x, y=input_y, res=input_res)
    c1.clear()
    assert np.array_equal(c1.x, expected_x)
    assert np.array_equal(c1.y, expected_y)
    assert np.array_equal(c1.data_order, expected_data_order)
    assert np.array_equal(c1.clusters, expected_clusters)
    assert c1.res == expected_res
    assert c1.current_idx == expected_current_index
    assert c1.lastcluster_idx == expected_last_cluster_idx
    assert c1.lastpoint_idx == expected_last_pt_idx
    assert c1.status == expected_status
