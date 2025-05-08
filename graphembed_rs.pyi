from __future__ import annotations
from typing import Optional

# ----------  Embedding  ----------
def embed_hope_rank(
    csv: str,
    symetric: bool,
    target_rank: int,
    nbiter: int,
    output: Optional[str] = None,
) -> None: ...
    """
    Compute a HOPE embedding using a target rank, rank algorithm in GSVD.

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    target_rank:
        Rank (number of singular vectors) to keep.
    nbiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    output:
        If given, write the embedding to *output* as a BSON file.
    """
def embed_hope_precision(
    csv: str,
    symetric: bool,
    epsil: float,
    maxrank: int,
    blockiter: int,
    output: Optional[str] = None,
) -> None: ...
    """
    Compute a HOPE embedding using a precision, range approximation algorithm in GSVD.

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    epsil:
        Precision to stop.
    maxrank:
        Rank (number of singular vectors) to keep.
    blockiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    output:
        If given, write the embedding to *output* as a BSON file.
    """
def embed_sketching(
    csv: str,
    symetric: bool,
    decay: float,
    dim: int,
    nbiter: int,
    output: Optional[str] = None,
) -> None: ...
    """
    Compute a node embedding using NodeSketch algorithm.

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    decay:
        Exponential decay rate.
    dim:
        Dimension of embedded vector.
    nbiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    output:
        If given, write the embedding to *output* as a BSON file.
    """
# ----------  Validation (returns mean AUC) ----------
def validate_hope_rank(
    csv: str,
    symetric: bool,
    target_rank: int,
    nbiter: int,
    nbpass: int = 1,
    skip_frac: float = 0.2,
    centric: bool = False,
) -> float: ...
    """
    Compute a HOPE embedding using a target rank, rank algorithm in GSVD and validate accuracy (AUC).

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    target_rank:
        Rank (number of singular vectors) to keep.
    nbiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    nbpass:
        Number of times to run the embedding for benchmarking.
    skip_frac:
        Fraction of edges to skip when computing AUC.
    centric:
        If ``True``, use a centric AUC method for link prediction evaluation (default: False).
    """
def validate_hope_precision(
    csv: str,
    symetric: bool,
    epsil: float,
    maxrank: int,
    blockiter: int,
    nbpass: int = 1,
    skip_frac: float = 0.2,
    centric: bool = False,
) -> float: ...
    """
    Compute a HOPE embedding using a precision, range approximation algorithm in GSVD and validate accuracy (AUC).

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    epsil:
        Precision to stop.
    maxrank:
        Rank (number of singular vectors) to keep.
    blockiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    nbpass:
        Number of times to run the embedding for benchmarking.
    skip_frac:
        Fraction of edges to skip when computing AUC.
    centric:
        If ``True``, use a centric AUC method for link prediction evaluation (default: False).
    """
def validate_sketching(
    csv: str,
    symetric: bool,
    decay: float,
    dim: int,
    nbiter: int,
    nbpass: int = 1,
    skip_frac: float = 0.2,
    centric: bool = False,
) -> float: ...
    """
    Compute a node embedding using NodeSketch algorithm and validate accuracy (AUC).

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    decay:
        Exponential decay rate.
    dim:
        Dimension of embedded vector.
    nbiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    nbpass:
        Number of times to run the embedding for benchmarking.
    skip_frac:
        Fraction of edges to skip when computing AUC.
    centric:
        If ``True``, use a centric AUC method for link prediction evaluation (default: False).  
    """

# ---------- VCMPR (precision/recall curves) ----------
def estimate_vcmpr_hope_rank(
    csv: str,
    symetric: bool,
    target_rank: int,
    nbiter: int,
    nbpass: int = 1,
    nb_edges: int = 10,
    skip_frac: float = 0.2,
) -> None: ...
    """
    Compute a HOPE embedding using a target rank, rank algorithm in GSVD and validate accuracy (VCMPR).

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    target_rank:
        Rank (number of singular vectors) to keep.
    nbiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    nb_edges:
        Number of edges around a node to check.
    nbpass:
        Number of times to run the embedding for benchmarking.
    skip_frac:
        Fraction of edges to skip when computing AUC.
    """
def estimate_vcmpr_sketching(
    csv: str,
    symetric: bool,
    decay: float,
    dim: int,
    nbiter: int,
    nbpass: int = 1,
    nb_edges: int = 10,
    skip_frac: float = 0.2,
) -> None: ...
    """
    Compute a node embedding using NodeSketch algorithm and validate accuracy (VCMPR).

    Parameters
    ----------
    csv:
        Path to the input edge list tabular file.
    decay:
        Exponential decay rate.
    dim:
        Dimension of embedded vector.
    nbiter:
        Number of iterations around a node, between 2 and 5.
    symetric:
        ``True`` for undirected graph/network; ``False`` for directed graphs/network.
    nb_edges:
        Number of edges around a node to check.
    nbpass:
        Number of times to run the embedding for benchmarking.
    skip_frac:
        Fraction of edges to skip when computing VCMPR.
    """
def load_embedding_bson(
    path: str | pathlib.Path,
    *,
    want_in: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]: ...