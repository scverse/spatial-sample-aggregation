import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import entropy
from squidpy._utils import NDArrayA


# TODO: this should go into squidpy/gr/_nhood.py
def _get_neighbor_counts(
    data: NDArrayA,
    indices: NDArrayA,
    indptr: NDArrayA,
    cats: NDArrayA,  # Array mapping cell indices to their types
    output: NDArrayA,  # Shape: (n_cells, n_celltypes)
) -> NDArrayA:
    indices_list = np.split(indices, indptr[1:-1])
    data_list = np.split(data, indptr[1:-1])
    for i in range(len(data_list)):  # Iterate over cells
        cur_row = i  # Each row corresponds to a cell
        cur_indices = indices_list[i]
        cur_data = data_list[i]
        for j, val in zip(cur_indices, cur_data, strict=False):
            cur_col = cats[j]  # Column corresponds to cell type
            output[cur_row, cur_col] += val
    return output


def get_neighbor_counts(adata, cluster_key="cell_type", connectivity_key="spatial_connectivities"):
    """Computes the number of each cell type in one-hop neighbors and stores it in adata.obsm['neighbor_counts']."""
    cats = adata.obs[cluster_key]
    mask = ~pd.isnull(cats).values
    cats = cats.loc[mask]
    if not len(cats):
        raise RuntimeError(f"After removing NaNs in `adata.obs[{cluster_key!r}]`, none remain.")

    g = adata.obsp[connectivity_key]
    g = g[mask, :][:, mask]
    n_cats = len(cats.cat.categories)

    g_data = np.broadcast_to(1, shape=len(g.data))
    dtype = int if pd.api.types.is_bool_dtype(g.dtype) or pd.api.types.is_integer_dtype(g.dtype) else float
    output: NDArrayA = np.zeros((n_cats, n_cats), dtype=dtype)

    return _get_neighbor_counts(g_data, g.indices, g.indptr, cats.cat.codes.to_numpy(), output)


def compute_node_feature(adata: AnnData, metric: str, connectivity_key: str, **kwargs) -> NDArrayA:
    """
    Compute a node-level feature based on the selected metric.

    Parameters
    ----------
    - adata: AnnData object
    - metric: str, the metric to compute ('shannon', 'degree', 'mean_distance')
    - connectivity_key: str, the key for the adjacency matrix in `adata.obsp`
    - kwargs: additional parameters for specific computations (e.g., `n_hops` for Shannon)

    Returns
    -------
    - np.ndarray: Node-level feature values indexed by cell ID
    """
    node_feature_functions = {
        "shannon": compute_shannon_diversity,
        "degree": calculate_degree,
        "mean_distance": calculate_mean_distance,
    }

    if metric not in node_feature_functions:
        raise ValueError(f"Unsupported metric: {metric}")

    return node_feature_functions[metric](adata, connectivity_key=connectivity_key, **kwargs).reshape(-1, 1)


def calculate_degree(adata: AnnData, connectivity_key: str = "radius_cut_connectivities", **kwargs) -> NDArrayA:
    """Compute the degree of each node."""
    return adata.obsp[connectivity_key].sum(axis=1)


def calculate_mean_distance(adata: AnnData, connectivity_key: str = "delaunay_distances", **kwargs) -> NDArrayA:
    """Compute the mean distance to neighbors."""
    return np.nanmean(adata.obsp[connectivity_key].toarray(), axis=1)


def compute_shannon_diversity(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    cluster_key: str = "cell_type",
    **kwargs,
) -> NDArrayA:
    """
    Compute Shannon diversity index for each node based on neighbor counts.

    Parameters
    ----------
    - adata: AnnData object
    - connectivity_key: str, key in adata.obsp corresponding to the adjacency matrix
    - cluster_key: str, column in adata.obs that contains categorical annotations (e.g., cell type)
    - kwargs: additional arguments (not used here but included for interface consistency)

    Returns
    -------
    - np.ndarray: Shannon diversity values indexed by cell ID
    """
    # Compute neighbor counts directly
    neighbor_counts = get_neighbor_counts(adata, cluster_key=cluster_key, connectivity_key=connectivity_key)

    # Normalize to probabilities
    probabilities = neighbor_counts / neighbor_counts.sum(axis=1, keepdims=True)

    # Compute Shannon diversity (entropy), ignoring zero probabilities
    shannon_diversity = np.apply_along_axis(lambda p: entropy(p[p > 0], base=2), 1, probabilities.values)

    return np.ndarray(shannon_diversity)


def aggregate_by_group(
    adata: AnnData,
    sample_key: str,
    node_feature_key: str,
    cluster_key: str | None = None,
    aggregation: str = "mean",
    key_added: str = "aggregated_features",
) -> None:
    """
    Aggregate node-level features by a sample group and optionally by annotation.

    Parameters
    ----------
    - adata: AnnData object
    - sample_key: str, column in `adata.obs` indicating the sample group
    - node_feature_key: str, column in `adata.obs` containing the node-level feature to aggregate
    - cluster_key: Optional[str], column in `adata.obs` for additional grouping (e.g., cell type)
    - aggregation: str, aggregation method ('mean', 'median', 'sum', None)
    - key_added: str, key under which results are stored in `adata.uns`

    Returns
    -------
    - None (Results are stored in `adata.uns[output_key]`)
    """
    if node_feature_key not in adata.obs.columns:
        raise ValueError(f"Column '{node_feature_key}' not found in adata.obs")

    if sample_key not in adata.obs.columns:
        raise ValueError(f"Column '{sample_key}' not found in adata.obs")

    if cluster_key and cluster_key not in adata.obs.columns:
        raise ValueError(f"Column '{cluster_key}' not found in adata.obs")

    # Select the aggregation function
    agg_methods = {
        "mean": "mean",
        "median": "median",
        "sum": "sum",
    }

    if aggregation is None:
        return

    if aggregation not in agg_methods:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    # Perform aggregation
    if cluster_key:
        aggregated = (
            adata.obs.groupby([sample_key, cluster_key])[node_feature_key]
            .agg(agg_methods[aggregation])
            .unstack()  # Pivot so that annotation_key values become columns
        )
    else:
        aggregated = adata.obs.groupby(sample_key)[node_feature_key].agg(agg_methods[aggregation])

    # TODO: adapt to squidpy save function
    adata.uns[key_added] = aggregated
