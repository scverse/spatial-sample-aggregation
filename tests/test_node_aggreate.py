import numpy as np
import pandas as pd
import pytest
import scipy
from anndata import AnnData
from spatial_sample_aggregation.tl.aggregate import aggregate_by_node
from spatial_sample_aggregation.tl.compute_node_features import get_neighbor_counts


@pytest.mark.parametrize("metric", ["shannon", "degree", "mean_distance"])
def test_aggregate_by_node(sample_adata, metric):
    """Test that aggregate_by_node correctly computes and stores metrics."""
    added_key = f"{metric}_aggregated"

    aggregate_by_node(
        adata=sample_adata,
        library_key="sample_id",
        cluster_key="cell_type",
        metric=metric,
        aggregation="mean",
        connectivity_key="spatial_connectivities",
        added_key=added_key,
    )

    # Check if the computed metric is stored in `adata.obs`
    assert added_key in sample_adata.obs, f"{added_key} was not stored in obs."

    # Check if the aggregated result is stored in `adata.uns`
    assert added_key in sample_adata.uns, f"{added_key} was not stored in uns."

    # Ensure aggregated values are not empty
    assert not sample_adata.obs[added_key].isna().all(), f"All {added_key} values are NaN."



def test_invalid_metric(sample_adata):
    """Ensure function raises an error for unsupported metrics."""
    with pytest.raises(ValueError):
        aggregate_by_node(
            adata=sample_adata,
            library_key="sample_id",
            cluster_key="cell_type",
            metric="invalid_metric",
            aggregation="mean",
            connectivity_key="spatial_connectivities",
        )

def test_missing_connectivity_key(sample_adata):
    """Ensure function raises an error if connectivity key is missing."""
    del sample_adata.obsp["spatial_connectivities"]

    with pytest.raises(KeyError):
        aggregate_by_node(
            adata=sample_adata,
            library_key="sample_id",
            cluster_key="cell_type",
            metric="shannon",
            aggregation="mean",
            connectivity_key="spatial_connectivities",
        )