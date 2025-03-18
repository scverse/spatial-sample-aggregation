import numpy as np
import pandas as pd
import pytest
import scipy
from anndata import AnnData

from spatial_sample_aggregation.tl import compute_node_feature


@pytest.mark.parametrize(
    "metric",
    [
        ("degree"),
        ("mean_distance"),
        ("shannon")
    ],
)
def test_compute_node_feature(sample_adata, metric):
    result = compute_node_feature(sample_adata, metric, connectivity_key="spatial_connectivities")

    # Check type
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray."

    # Check shape
    assert result.shape == (sample_adata.n_obs,1), f"Expected shape {(sample_adata.n_obs,1)}, but got {result.shape}."

    # Check no NaNs in degree and mean_distance (shannon may have NaNs for isolated nodes)
    if metric in ["degree", "mean_distance"]:
        assert not np.isnan(result).any(), f"NaN values found in {metric} computation."


@pytest.mark.parametrize("invalid_metric", ["non_existent_metric", "random", "graph_density"])
def test_compute_node_feature_invalid_metric(sample_adata, invalid_metric):
    with pytest.raises(ValueError, match=f"Unsupported metric: {invalid_metric}"):
        compute_node_feature(sample_adata, invalid_metric, connectivity_key="spatial_connectivities")
