import pandas as pd
from anndata import AnnData


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
    adata, *, sample_key: str, annotation_key: str, metric: str = "shannon", aggregate_by: str = "mean"
) -> pd.DataFrame:
    """
    Compute a metric on every node of the neighborhood graph. Then aggregate this metric by a group (e.g. cell-type).

    Parameters
    ----------
    metric
        possible metrics are shannon entropy, count (-> get percentage of niches/cell-types), ... (?)
    """
    pass
