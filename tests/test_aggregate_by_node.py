import pytest
import numpy as np

from spatial_sample_aggregation.tl.aggregate import aggregate_by_node
from spatial_sample_aggregation.tl import compute_node_feature, get_neighbor_counts


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


def test_get_neighbor_counts(sample_adata):
    """Functional test to see if the result matrix has the correct shape."""
    cell_by_celltype_matrix = get_neighbor_counts(sample_adata)
    assert cell_by_celltype_matrix.shape == (sample_adata.obs.shape[0], len(sample_adata.obs["cell_type"].unique()))
    assert not np.isnan(cell_by_celltype_matrix).any(), f"Some values are NaN. Matrix:\n{cell_by_celltype_matrix}"


def test_get_neighbor_counts(sample_adata):
    """Functional test to see if the result matrix has the correct shape."""
    cell_by_celltype_matrix = get_neighbor_counts(sample_adata)
    assert cell_by_celltype_matrix.shape == (sample_adata.obs.shape[0], len(sample_adata.obs["cell_type"].unique()))
    assert not np.isnan(cell_by_celltype_matrix).any(), f"Some values are NaN. Matrix:\n{cell_by_celltype_matrix}"
    assert "composition_matrix" in sample_adata.obsm, "composition_matrix not found in adata.obsm."
    assert sample_adata.obsm["composition_matrix"].shape == cell_by_celltype_matrix.shape, (
        "composition_matrix shape does not match cell_by_celltype_matrix shape."
    )


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
