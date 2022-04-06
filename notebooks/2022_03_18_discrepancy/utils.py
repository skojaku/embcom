import glob
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import graph_tool.all as gt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_files(dirname, dtypes=None):
    if isinstance(dirname, str):
        input_files = list(glob.glob(dirname + "/*"))
    else:
        input_files = dirname

    def get_params(filenames):
        def _get_params(filename, sep="~"):
            params = pathlib.Path(filename).stem.split("_")
            retval = {"filename": filename}
            for p in params:
                if sep not in p:
                    continue
                kv = p.split(sep)

                retval[kv[0]] = kv[1]
            return retval

        return pd.DataFrame([_get_params(filename) for filename in filenames])

    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")

    if dtypes is not None:
        for k, v in dtypes.items():
            if v == "int":
                df[k] = df[k].astype(float)
            df[k] = df[k].astype(v)
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


def get_membership(n, q):
    return np.sort(np.arange(n) % q)


def generate_network(Cave, mixing_rate, N, q, memberships=None, **params):

    if memberships is None:
        memberships = get_membership(N, q)

    q = int(np.max(memberships) + 1)
    N = len(memberships)
    U = sparse.csr_matrix((np.ones(N), (np.arange(N), memberships)), shape=(N, q))
    Cin, Cout = get_cin_cout(Cave, mixing_rate, q)
    pout = Cout / N
    pin = Cin / N

    Nk = np.array(U.sum(axis=0)).reshape(-1)

    P = np.ones((q, q)) * pout + np.eye(q) * (pin - pout)
    probs = np.diag(Nk) @ P @ np.diag(Nk)

    gt_params = {
        "b": memberships,
        "probs": probs,
    }

    # Generate the network until the degree sequence
    # satisfied the thresholds
    while True:
        g = gt.generate_sbm(**gt_params)
        A = gt.adjacency(g).T
        A.data = np.ones_like(A.data)
        # check if the graph is connected
        if connected_components(A)[0] == 1:
            break
    return A, memberships


def make_node2vec_matrix(A, L, returnPt=False):
    n = A.shape[0]
    deg = np.array(A.sum(axis=0)).reshape(-1)
    P = sparse.diags(1 / np.maximum(deg, 1)) @ A
    pi = deg / np.sum(deg)
    logpi = np.log(pi)
    Prwr = calc_rwr(P.toarray(), L, returnPt=returnPt)
    R = np.log(Prwr) - np.outer(np.ones(n), logpi)
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = 0
    return R


def get_cin_cout(Cave, mixing_rate, q, **params):
    cout = Cave * mixing_rate
    cin = q * Cave - (q - 1) * cout
    return cin, cout


def make_node2vec_matrix_limit(A, L, returnPt=False):
    deg = np.array(A.sum(axis=0)).reshape(-1)
    P = sparse.diags(1 / np.maximum(deg, 1)) @ A
    Prwr = calc_rwr(P.toarray(), L, returnPt=returnPt)
    R = np.sum(deg) * Prwr @ sparse.diags(1 / np.maximum(deg, 1)) - 1

    return R


def calc_rwr(P, L, returnPt=False):

    Pt = P.copy()
    Ps = None
    for t in range(L):
        if Ps is None:
            Ps = Pt / L
            continue
        Pt = P @ Pt
        Ps += Pt / L
    if returnPt:
        return Pt
    return Ps


def graph_kernel(X, phi):
    s, v = np.linalg.eig(X)
    s = phi(s)
    return v @ np.diag(s) @ v.T


def calc_esim(y, ypred):
    """Element centric similarity."""
    ylab, y = np.unique(y, return_inverse=True)
    ypredlab, ypred = np.unique(ypred, return_inverse=True)

    Ka, Kb = len(ylab), len(ypredlab)

    K = int(np.maximum(np.max(y), np.max(ypred))) + 1
    M = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(M, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(M, K)
    )

    fA = np.array(UA.sum(axis=0)).reshape(-1)
    fB = np.array(UB.sum(axis=0)).reshape(-1)

    fAB = UA.T @ UB

    S = 0
    for i in range(len(y)):
        S += (fAB[y[i], ypred[i]] ** 2) * np.minimum(
            1 / np.maximum(1, fA[y[i]]), 1 / np.maximum(1, fB[ypred[i]])
        )
    S /= M

    Srand = np.minimum(1 / Ka, 1 / Kb)

    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected
