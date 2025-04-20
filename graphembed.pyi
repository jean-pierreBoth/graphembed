from typing import Optional

# ----------  Embedding  ----------
def embed_hope_rank(
    csv: str,
    symetric: bool,
    target_rank: int,
    nbiter: int,
    output: Optional[str] = None,
) -> None: ...
def embed_hope_precision(
    csv: str,
    symetric: bool,
    epsil: float,
    maxrank: int,
    blockiter: int,
    output: Optional[str] = None,
) -> None: ...
def embed_sketching(
    csv: str,
    symetric: bool,
    decay: float,
    dim: int,
    nbiter: int,
    output: Optional[str] = None,
) -> None: ...

# ----------  Validation (returns mean AUC) ----------
def validate_hope_rank(
    csv: str,
    symetric: bool,
    target_rank: int,
    nbiter: int,
    nbpass: int = 10,
    skip_frac: float = 0.1,
    centric: bool = False,
) -> float: ...
def validate_hope_precision(
    csv: str,
    symetric: bool,
    epsil: float,
    maxrank: int,
    blockiter: int,
    nbpass: int = 10,
    skip_frac: float = 0.1,
    centric: bool = False,
) -> float: ...
def validate_sketching(
    csv: str,
    symetric: bool,
    decay: float,
    dim: int,
    nbiter: int,
    nbpass: int = 10,
    skip_frac: float = 0.1,
    centric: bool = False,
) -> float: ...

# ---------- VCMPR (precision/recall curves) ----------
def estimate_vcmpr_hope_rank(
    csv: str,
    symetric: bool,
    target_rank: int,
    nbiter: int,
    nbpass: int = 2,
    topk: int = 10,
    skip_frac: float = 0.1,
) -> None: ...
def estimate_vcmpr_sketching(
    csv: str,
    symetric: bool,
    decay: float,
    dim: int,
    nbiter: int,
    nbpass: int = 2,
    topk: int = 10,
    skip_frac: float = 0.1,
) -> None: ...

