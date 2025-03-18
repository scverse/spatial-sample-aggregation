import numpy as np
import pandas as pd
import pytest
import scipy
from anndata import AnnData
from spatial_sample_aggregation.tl.aggregate import aggregate_by_node
from spatial_sample_aggregation.tl.compute_node_features import get_neighbor_counts

@pytest.fixture
def sample_adata():
    """Creates a small AnnData object for testing."""
    # Create observation dataframe with 2 samples and 10 cells each (total 20 cells)
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(20)],
            "cell_type": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"] * 2,  # Repeating for two samples
            "sample_id": ["S1"] * 10 + ["S2"] * 10,  # First 10 cells in S1, next 10 in S2
        }
    ).set_index("cell_id")

    # Create an adjacency matrix with 2 separate connected components (one per sample)
    adjacency_matrix = scipy.sparse.block_diag(
        [
            scipy.sparse.csr_matrix(
                [
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],  # cell_0 connects to cell_1, cell_3, cell_9
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # cell_1 connects to cell_0, cell_2, cell_4
                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # cell_2 connects to cell_1, cell_3, cell_5
                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # cell_3 connects to cell_0, cell_2, cell_6
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  # cell_4 connects to cell_1, cell_5, cell_7
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # cell_5 connects to cell_2, cell_4, cell_6, cell_8
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # cell_6 connects to cell_3, cell_5, cell_7
                    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],  # cell_7 connects to cell_4, cell_6, cell_8
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # cell_8 connects to cell_5, cell_7, cell_9
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # cell_9 connects to cell_0, cell_8
                ]
            ),
            scipy.sparse.csr_matrix(
                [
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],  # cell_10 connects to cell_11, cell_13, cell_19
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # cell_11 connects to cell_10, cell_12, cell_14
                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # cell_12 connects to cell_11, cell_13, cell_15
                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # cell_13 connects to cell_10, cell_12, cell_16
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  # cell_14 connects to cell_11, cell_15, cell_17
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # cell_15 connects to cell_12, cell_14, cell_16, cell_18
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # cell_16 connects to cell_13, cell_15, cell_17
                    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],  # cell_17 connects to cell_14, cell_16, cell_18
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # cell_18 connects to cell_15, cell_17, cell_19
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # cell_19 connects to cell_10, cell_18
                ]
            ),
        ]
    )

    # Create AnnData object
    adata = AnnData(obs=obs)
    adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obsm["spatial"] = np.random.rand(20, 2)
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    return adata

@pytest.mark.parametrize("metric", ["shannon", "degree", "mean_distance"])
def test_aggregate_by_node(sample_adata, metric):
    """Test that aggregate_by_node correctly computes and stores metrics."""
    added_key = f"{metric}_aggregated"

    aggregate_by_node(
        adata=sample_adata,
        sample_key="sample_id",
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
            sample_key="sample_id",
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
            sample_key="sample_id",
            cluster_key="cell_type",
            metric="shannon",
            aggregation="mean",
            connectivity_key="spatial_connectivities",
        )