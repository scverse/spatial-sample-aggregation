import pandas as pd
from anndata import AnnData
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import _assert_categorical_obs, _assert_connectivity_key

from .compute_node_features import aggregate_by_group, compute_node_feature


def aggregate_by_edge(
    adata: AnnData, library_key: str, annotation_key: str, use_edge_weight: bool = False
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
    library_key: str,
    cluster_key: str = None,  # TODO: annotation_key --> cluster_key to adapt to squidpy notation
    metric: str = "shannon",
    aggregation: str = "mean",  # TODO: new parameter --> check squidpy
    connectivity_key: str = "spatial_connectivities",  # TODO: new parameter
    key_added: str = None,
    **kwargs,
) -> None:
    """
    Compute a node-level metric and aggregate it by a sample group.

    Parameters
    ----------
    - adata: AnnData, input data
    - library_key: str, column in `adata.obs` to group by
    - cluster_key: Optional[str], cell type or similar annotation
    - metric: str, metric to compute ('shannon', 'degree', 'mean_distance')
    - aggregation: str, aggregation method ('mean', 'median', 'sum', 'none')
    - connectivity_key: str, adjacency matrix key
    - key_added: Optional[str], key under which aggregated results are stored in `adata.uns`. Defaults to `metric`.
    - kwargs: Additional parameters passed to metric computation functions.

    Returns
    -------
    - None (Results are stored in `adata.obs[key_added]` and the agggregated features are added in `adata.uns[key_added]` if aggregation is not None)
    """
    # Determine where to store the results (default to metric name)
    if key_added is None:
        key_added = metric

    # TODO: adapt to squidpy: connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    # Compute node-level feature
    node_features = compute_node_feature(
        adata, metric, connectivity_key=connectivity_key, cluster_key=cluster_key, library_key=library_key, **kwargs
    )

    # TODO: adapt to squidpy gr_utils _save_data(adata, attr="obs", key=Key.obs.feature(feature_column), data=node_features)
    adata.obs[key_added] = node_features  # TODO: store in obs here or in the indivdiual functions

    # Aggregate the computed metric at the sample level
    aggregate_by_group(
        adata,
        library_key=library_key,
        node_feature_key=key_added,
        cluster_key=cluster_key,
        aggregation=aggregation,
        key_added=key_added,
    )
