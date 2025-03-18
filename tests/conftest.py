import pytest
import pandas as pd
import numpy as np
import scipy.sparse
from anndata import AnnData

@pytest.fixture
def sample_adata():
    """Creates a small AnnData object for testing."""
    # Create observation dataframe with 2 samples and 10 cells each (total 20 cells)
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(20)],
            "cell_type": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"] * 2,  # Repeating for two samples
            "sample_id": ["S1"] * 10 + ["S2"] * 10,  # First 10 cells in S1, next 10 in S2
            "node_feature": np.random.rand(20)
        }
    ).set_index("cell_id")

    # Create an adjacency matrix with 2 separate connected components (one per sample)
    adjacency_matrix = scipy.sparse.block_diag(
        [
            scipy.sparse.csr_matrix(
                [
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],  
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  
                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  
                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],  
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  
                    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],  
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],  
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
                ]
            ),
            scipy.sparse.csr_matrix(
                [
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],  
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  
                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  
                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  
                    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],  
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  
                    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],  
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],  
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  
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
