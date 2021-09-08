import glob
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def get_params(filenames):
    return pd.DataFrame([_get_params(filename) for filename in filenames])


def _get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "=" not in p:
            continue
        kv = p.split("=")

        retval[kv[0]] = kv[1]
    return retval


def load_files(dirname):
    input_files = list(glob.glob(dirname + "/*"))
    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


def row_normalize(mat, mode="prob"):
    """Normalize a sparse CSR matrix row-wise (each row sums to 1) If a row is
    all 0's, it remains all 0's.

    Parameters
    ----------
    mat : scipy.sparse.csr matrix
        Matrix in CSR sparse format
    Returns
    -------
    out : scipy.sparse.csr matrix
        Normalized matrix in CSR sparse format
    """
    if mode == "prob":
        denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
        return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat
    elif mode == "norm":
        denom = np.sqrt(np.array(mat.multiply(mat).sum(axis=1)).reshape(-1))
        return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat
    return np.nan
