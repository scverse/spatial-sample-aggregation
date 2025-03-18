import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import entropy


def get_n_hop_neighbors(G, cell_idx, n_hops):
    """Find all nodes within n_hops from cell_idx using BFS."""
    visited = set(cell_idx)  # Track visited nodes
    current_level = {cell_idx}  # Start with the original cell

    for _ in range(n_hops):
        next_level = set()  # Store neighbors at the next hop level
        for node in current_level:
            next_level.update(set(G.neighbors(node)) - visited)  # Avoid revisiting
        if not next_level:
            break  # No more neighbors to explore
        visited.update(next_level)  # Mark them as visited
        current_level = next_level  # Move to the next hop

    visited.remove(cell_idx)  # Exclude the original node
    return visited


def get_neighbor_counts(adata, n_hops=1, phenotype_col="celllineage", graph_key="generic_connectivities"):
    """Computes the number of each cell type in one-hop neighbors and stores it in adata.obsm['neighbor_counts']."""
    # Get unique cell types
    cell_types = adata.obs[phenotype_col].unique()

    # Create an empty dataframe
    neighbor_counts = pd.DataFrame(
        np.zeros((adata.n_obs, len(cell_types)), dtype=int), index=adata.obs_names, columns=cell_types
    )

    adjacency_matrix = adata.obsp[graph_key]

    # Convert to NetworkX graph
    G = nx.from_scipy_sparse_array(adjacency_matrix)

    # Iterate over each cell
    for cell_idx in range(adata.n_obs):
        neighbors = get_n_hop_neighbors(G, cell_idx, n_hops)

        # Get their cell types
        neighbor_types = adata.obs.iloc[list(neighbors)][phenotype_col]

        # Count occurrences
        neighbor_counts.iloc[cell_idx] = neighbor_types.value_counts().reindex(cell_types, fill_value=0)

    # Store in AnnData object
    return neighbor_counts


def compute_node_feature(adata: AnnData, metric: str, connectivity_key: str, **kwargs) -> pd.Series:
    """
    Compute a node-level feature based on the selected metric.

    Parameters
    ----------
    - adata: AnnData object
    - metric: str, the metric to compute ('shannon', 'degree', 'mean_distance')
    - graph_key: str, the key for the adjacency matrix in `adata.obsp`
    - kwargs: additional parameters for specific computations (e.g., `n_hops` for Shannon)

    Returns
    -------
    - pd.Series: Node-level feature values indexed by cell ID
    """
    node_feature_functions = {
        "shannon": compute_shannon_diversity,
        "degree": calculate_degree,
        "mean_distance": calculate_mean_distance,
    }

    if metric not in node_feature_functions:
        raise ValueError(f"Unsupported metric: {metric}")

    return node_feature_functions[metric](adata, graph_key=connectivity_key, **kwargs)


def calculate_degree(adata: AnnData, graph_key: str = "radius_cut_connectivities", **kwargs) -> pd.Series:
    """Compute the degree of each node."""
    return adata.obsp[graph_key].sum(axis=1)


def calculate_mean_distance(adata: AnnData, graph_key: str = "delaunay_distances", **kwargs) -> pd.Series:
    """Compute the mean distance to neighbors."""
    return np.nanmean(adata.obsp[graph_key].toarray(), axis=1)


def compute_shannon_diversity(
    adata: AnnData,
    graph_key: str = "generic_connectivities",
    n_hops: int = 1,
    phenotype_col: str = "celllineage",
    **kwargs,
) -> pd.Series:
    """
    Compute Shannon diversity index for each node based on neighbor counts.

    Parameters
    ----------
    - adata: AnnData object
    - graph_key: str, key in adata.obsp corresponding to the adjacency matrix
    - n_hops: int, number of hops to consider for neighbors
    - phenotype_col: str, column in adata.obs that contains categorical annotations (e.g., cell type)
    - kwargs: additional arguments (not used here but included for interface consistency)

    Returns
    -------
    - pd.Series: Shannon diversity values indexed by cell ID
    """
    # Compute neighbor counts directly
    neighbor_counts = get_neighbor_counts(adata, phenotype_col=phenotype_col, graph_key=graph_key, n_hops=n_hops)

    # Normalize to probabilities
    probabilities = neighbor_counts / neighbor_counts.sum(axis=1, keepdims=True)

    # Compute Shannon diversity (entropy), ignoring zero probabilities
    shannon_diversity = np.apply_along_axis(lambda p: entropy(p[p > 0], base=2), 1, probabilities.values)

    return shannon_diversity


def aggregate_by_group(
    adata: AnnData,
    sample_key: str,
    node_feature_key: str,
    cluster_key: str = None,
    aggregation: str = "mean",
    added_key: str = "aggregated_features",
) -> None:
    """
    Aggregate node-level features by a sample group and optionally by annotation.

    Parameters
    ----------
    - adata: AnnData object
    - sample_key: str, column in `adata.obs` indicating the sample group
    - node_feature_key: str, column in `adata.obs` containing the node-level feature to aggregate
    - annotation_key: Optional[str], column in `adata.obs` for additional grouping (e.g., cell type)
    - aggregation_function: str, aggregation method ('mean', 'median', 'sum', 'none')
    - output_key: str, key under which results are stored in `adata.uns`

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
        "none": lambda x: x,  # No aggregation, keeps raw values
    }

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

    # Store results in adata.uns
    adata.uns[added_key] = aggregated
