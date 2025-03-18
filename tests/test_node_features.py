import numpy as np
import pandas as pd
import pytest
import scipy
from anndata import AnnData

from spatial_sample_aggregation.tl import compute_node_feature


@pytest.fixture
def adata():
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


@pytest.mark.parametrize(
    "metric",
    [
        ("degree"),
        ("mean_distance"),
        ("shannon")
    ],
)
def test_compute_node_feature(adata, metric):
    result = compute_node_feature(adata, metric, connectivity_key="spatial_connectivities")

    # Check type
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray."

    # Check shape
    assert result.shape == (adata.n_obs,1), f"Expected shape {(adata.n_obs,1)}, but got {result.shape}."

    # Check no NaNs in degree and mean_distance (shannon may have NaNs for isolated nodes)
    if metric in ["degree", "mean_distance"]:
        assert not np.isnan(result).any(), f"NaN values found in {metric} computation."


@pytest.mark.parametrize("invalid_metric", ["non_existent_metric", "random", "graph_density"])
def test_compute_node_feature_invalid_metric(adata, invalid_metric):
    with pytest.raises(ValueError, match=f"Unsupported metric: {invalid_metric}"):
        compute_node_feature(adata, invalid_metric, connectivity_key="spatial_connectivities")
