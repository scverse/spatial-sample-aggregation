import numpy as np
import pandas as pd
import pytest

from spatial_sample_aggregation.tl import aggregate_by_group, compute_node_feature, get_neighbor_counts


@pytest.mark.parametrize(
    "metric",
    [("degree"), ("mean_distance"), ("shannon")],
)
def test_compute_node_feature(sample_adata, metric):
    result = compute_node_feature(sample_adata, metric, connectivity_key="spatial_connectivities")

    # Check type
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray."

    # Check shape
    assert result.shape == (sample_adata.n_obs, 1), f"Expected shape {(sample_adata.n_obs, 1)}, but got {result.shape}."

    # Check no NaNs in degree and mean_distance (shannon may have NaNs for isolated nodes)
    if metric in ["degree", "mean_distance"]:
        assert not np.isnan(result).any(), f"NaN values found in {metric} computation."


@pytest.mark.parametrize("invalid_metric", ["non_existent_metric", "random", "graph_density"])
def test_compute_node_feature_invalid_metric(sample_adata, invalid_metric):
    with pytest.raises(ValueError, match=f"Unsupported metric: {invalid_metric}"):
        compute_node_feature(sample_adata, invalid_metric, connectivity_key="spatial_connectivities")


@pytest.mark.parametrize("aggregation", ["mean", "median", "sum"])
def test_aggregate_by_group(sample_adata, aggregation):
    aggregate_by_group(
        sample_adata,
        library_key="sample_id",
        node_feature_key="node_feature",
        aggregation=aggregation,
        key_added="aggregated_features",
    )

    # Check that the aggregated results are stored in `adata.uns`
    assert "aggregated_features" in sample_adata.uns, "Aggregated features not found in adata.uns."

    aggregated = sample_adata.uns["aggregated_features"]

    # Check that aggregation returns a DataFrame
    assert isinstance(aggregated, pd.Series) or isinstance(aggregated, pd.DataFrame), (
        "Aggregation output should be Series or DataFrame."
    )

    # Check that the aggregated index matches unique sample keys
    assert set(aggregated.index) == set(sample_adata.obs["sample_id"].unique()), (
        "Aggregated index does not match sample keys."
    )


@pytest.mark.parametrize("invalid_aggregation", ["invalid_method", "average", "total"])
def test_aggregate_by_group_invalid_aggregation(sample_adata, invalid_aggregation):
    with pytest.raises(ValueError, match=f"Unsupported aggregation method: {invalid_aggregation}"):
        aggregate_by_group(
            sample_adata,
            library_key="sample_id",
            node_feature_key="node_feature",
            aggregation=invalid_aggregation,
            key_added="aggregated_features",
        )


@pytest.mark.parametrize("missing_key", ["missing_sample", "missing_feature"])
def test_aggregate_by_group_missing_keys(sample_adata, missing_key):
    library_key = "sample_id" if missing_key != "missing_sample" else "non_existent_column"
    node_feature_key = "node_feature" if missing_key != "missing_feature" else "non_existent_feature"

    with pytest.raises(ValueError, match="Column '.*' not found in adata.obs"):
        aggregate_by_group(
            sample_adata,
            library_key=library_key,
            node_feature_key=node_feature_key,
            aggregation="mean",
            key_added="aggregated_features",
        )


def test_aggregate_by_group_none_aggregation(sample_adata):
    aggregate_by_group(
        sample_adata,
        library_key="sample_id",
        node_feature_key="node_feature",
        aggregation=None,
        key_added="aggregated_features",
    )

    # Check that nothing is written to `adata.uns`
    assert "aggregated_features" not in sample_adata.uns, "No aggregation should be written when aggregation=None."


def test_get_neighbor_counts(sample_adata):
    """Functional test to see if the result matrix has the correct shape."""
    cell_by_celltype_matrix = get_neighbor_counts(sample_adata)
    assert cell_by_celltype_matrix.shape == (sample_adata.obs.shape[0], len(sample_adata.obs["cell_type"].unique()))
    assert not np.isnan(cell_by_celltype_matrix).any(), f"Some values are NaN. Matrix:\n{cell_by_celltype_matrix}"
