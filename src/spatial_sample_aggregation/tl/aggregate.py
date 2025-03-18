import pandas as pd
from anndata import AnnData
from compute_node_features import aggregate_by_group, compute_node_feature
from squidpy._constants._pkg_constants import Key
from squidpy.gr_utils import _assert_categorical_obs, _assert_connectivity_key


def aggregate_by_edge(
    adata: AnnData, sample_key: str, annotation_key: str, use_edge_weight: bool = False
) -> pd.DataFrame:
    """
    Aggregate spatial neighborhood graph taking into account neighbors

    For each pair of cell-types (niches) count the number of edges in the spatial neighborhood graph connecting
    these two cell-types (niches) in each sample. Normalize the edge count to sum to 1 for each sample.
    """
    pass


def aggregate_by_node(
    adata: AnnData,
    *,
    sample_key: str,
    cluster_key: str = None,  # TODO: annotation_key --> cluster_key to adapt to squidpy notation
    metric: str = "shannon",
    aggregation: str = "mean",  # TODO: new parameter --> check squidpy
    connectivity_key: str = "spatial_connectivities",  # TODO: new parameter
    added_key: str = None,
    n_hops: int = 1,
    **kwargs,
) -> None:
    """
    Compute a node-level metric and aggregate it by a sample group.

    Parameters
    ----------
    - adata: AnnData, input data
    - sample_key: str, column in `adata.obs` to group by
    - annotation_key: Optional[str], cell type or similar annotation
    - metric: str, metric to compute ('shannon', 'degree', 'mean_distance')
    - aggregate_by: str, aggregation method ('mean', 'median', 'sum', 'none')
    - graph_key: str, adjacency matrix key
    - added_key: Optional[str], key under which aggregated results are stored in `adata.uns`. Defaults to `metric`.
    - n_hops: int, number of hops for neighborhood metrics
    - kwargs: Additional parameters passed to metric computation functions.

    Returns
    -------
    - None (Results are stored in `adata.uns[added_key]`)
    """
    # Determine where to store the results (default to metric name)
    if added_key is None:
        added_key = metric

    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    # Compute node-level feature
    node_features = compute_node_feature(
        adata, metric, connectivity_key=connectivity_key, n_hops=n_hops, phenotype_col=cluster_key, **kwargs
    )

    # TODO: adapt to squidpy gr_utils _save_data(adata, attr="obs", key=Key.obs.feature(feature_column), data=node_features)
    adata.obs[added_key] = node_features  # TODO: store in obs here or in the indivdiual functions

    # Aggregate the computed metric at the sample level
    aggregate_by_group(
        adata,
        sample_key=sample_key,
        node_feature_key=added_key,
        cluster_key=cluster_key,
        aggregation=aggregation,
        added_key=added_key,
    )
