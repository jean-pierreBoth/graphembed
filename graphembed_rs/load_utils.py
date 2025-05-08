import numpy as np
from pathlib import Path
### install pymongo with pip install pymongo
# -- pymongo is a BSON decoder, not a BSON encoder --
from bson import decode_file_iter


def _dtype_from_type_name(tname: str):
    """Map the Rust type-name string to a NumPy dtype."""
    if tname == "f32":
        return np.float32
    if tname == "f64":
        return np.float64
    if tname == "usize":                 # stored as i64 in BSON
        return np.int64
    raise ValueError(f"Unknown type_name {tname!r} in BSON header")


def load_embedding_bson(path: str | Path, *, want_in: bool = False):
    """
    Parameters
    ----------
    path : str | pathlib.Path
        File written by graphembed::io::bson_dump(...)
    want_in : bool, default False
        • False … always return the OUT/source embedding (shape = (n, d))  
        • True  … additionally return the IN/target embedding when
          the dump is *asymmetric* (tuple(out, in_)).  For symmetric
          dumps the second item is None.

    Returns
    -------
    np.ndarray            (sym. dump or want_in=False)
    OR (out_emb, in_emb)  (want_in=True)
    """
    path = Path(path)
    with path.open("rb") as fh:
        docs = decode_file_iter(fh)

        # -- header ----------------------------------------------------
        header = next(docs)["header"]
        n      = int(header["nbdata"])
        d      = int(header["dimension"])
        sym    = bool(header["symetric"])
        dtype  = _dtype_from_type_name(header["type_name"])
        out_emb = np.empty((n, d), dtype=dtype)
        in_emb  = None if sym or not want_in else np.empty((n, d), dtype=dtype)

        # --OUT part --------------------------------------------------
        for _ in range(n):
            doc = next(docs)
            key, vec = next(iter(doc.items()))            # only 1 (key,val)
            idx, tag = map(int, key.split(","))
            assert tag == 0, f"expected tag 0, got {tag}"
            out_emb[idx] = np.asarray(vec, dtype=dtype)

        # -- IN part (if any) -----------------------------------------
        if not sym:
            for _ in range(n):
                doc = next(docs)
                key, vec = next(iter(doc.items()))
                idx, tag = map(int, key.split(","))
                assert tag == 1, f"expected tag 1, got {tag}"
                if in_emb is not None:                    # want_in == True
                    in_emb[idx] = np.asarray(vec, dtype=dtype)
                # else: silently drop it

        # -- optional indexation doc – skip for now --------------------
        # (decode_file_iter stops automatically at EOF)

    return (out_emb, in_emb) if want_in else out_emb
