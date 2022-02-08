"""Sample a network and sample the residual matrics."""
# %%
import itertools
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
from tqdm import tqdm

if "snakemake" in sys.modules:
    cin = int(snakemake.params["cin"])
    cout = int(snakemake.params["cout"])
    num_sample = int(snakemake.params["num_sample"])
    output_file = snakemake.output["output_file"]
else:
    output_file = "../data/"
    cin, cout = 30, 5
    num_sample = 30

#
# Data
#
def generate_dcSBM(cin, cout, n):
    pin, pout = cin / n, cout / n
    N = 2 * n
    gids = np.concatenate([np.zeros(n), np.ones(n)])
    d = (cin + cout) / 2

    within_edges = set([])
    target_num = stats.binom.rvs(int(n * (n - 1) / 2 * 2), pin, size=1)[0]
    esize = 0
    while esize < target_num:
        r = np.random.choice(2 * n, target_num - esize)
        c = np.random.choice(2 * n, target_num - esize)
        s = (gids[r] == gids[c]) * (r != c)
        r, c = r[s], c[s]
        eids = set(r + c * N)
        within_edges = within_edges.union(eids)
        esize = len(within_edges)

    between_edges = set([])
    target_num = stats.binom.rvs(n * n, pout, size=1)[0]
    esize = 0
    while esize < target_num:
        r = np.random.choice(2 * n, target_num - esize)
        c = np.random.choice(2 * n, target_num - esize)
        s = (gids[r] != gids[c]) * (r != c)
        r, c = r[s], c[s]
        eids = set(r + c * N)
        between_edges = between_edges.union(eids)
        esize = len(between_edges)

    edges = np.array(list(between_edges) + list(within_edges))
    r, c = divmod(edges, N)
    A = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(N, N))
    A = A + A.T
    A.data = np.ones_like(A.data)

    return A


def calc_matrix_P(A, T):
    deg = np.array(A.sum(axis=1)).reshape(-1)
    P = sparse.diags(1 / np.maximum(deg, 1)) @ A

    Pt = P.copy()
    for t in range(T - 1):
        Pt = Pt @ P + P
    Pt /= T
    return Pt.toarray()


def calc_matrix_R(A, T):
    deg = np.array(A.sum(axis=1)).reshape(-1)
    pi = deg / np.sum(deg)
    Pt = calc_matrix_P(A, T)
    N = len(deg)
    Pt = np.log(np.maximum(Pt, 1e-12))
    R = Pt - np.ones((N, 1)) @ np.log(pi).reshape((1, -1))
    return R


def sample_within_between(R, sample_num):
    n = int(R.shape[0] / 2)
    Rwithin = R[:n, :][:, :n].reshape(-1)
    Rbetween = R[n:, :][:, :n].reshape(-1)
    # return [np.mean(Rwithin)], [np.mean(Rbetween)]
    return np.random.choice(Rwithin, sample_num), np.random.choice(Rbetween, sample_num)


def predicted_pij(cin, cout, n, T):
    N = 2 * n

    Lambda = sparse.csr_matrix(np.array([[cin, cout], [cout, cin]]) / (cin + cout))
    Lamt = calc_matrix_P(Lambda.copy(), T)
    pcin = 2 * Lamt[0, 0] / N
    pcout = 2 * Lamt[0, 1] / N

    Lambda = Lambda.toarray()
    Lamt = Lambda.copy()
    pcin_sig = 0
    pcout_sig = 0
    d = (cin + cout) / 2
    for t in range(1, T + 1):
        dt = np.power(d, t)
        pcin_sig += 2 * Lamt[0, 0] / (N * dt)
        pcout_sig += 2 * Lamt[0, 1] / (N * dt)
        Lamt = Lamt @ Lambda
    pcin_sig /= T * T
    pcout_sig /= T * T
    return pcin, pcout, pcin_sig, pcout_sig


nlist = [100, 200, 400, 800, 1600]
Tlist = [3, 5, 10, 20]

dflist = []
for n, T, i in tqdm(list(itertools.product(nlist, Tlist, list(range(num_sample))))):
    A = generate_dcSBM(cin, cout, n)
    R = calc_matrix_P(A, T)
    rin, rout = sample_within_between(R, 1)
    df1 = pd.DataFrame({"r": rin, "T": T, "rtype": "sim-in", "n": n, "rvar": rin})
    df2 = pd.DataFrame({"r": rout, "T": T, "rtype": "sim-out", "n": n, "rvar": rout})
    dflist += [pd.concat([df1, df2])]

for n, T in tqdm(list(itertools.product(nlist, Tlist))):
    pcin, pcout, pcin_sig, pcout_sig = predicted_pij(cin, cout, n, T)
    dg = pd.DataFrame(
        {
            "r": [pcin, pcout],
            "n": [n, n],
            "T": [T, T],
            "rtype": ["in", "out"],
            "rvar": [pcin_sig, pcout_sig],
        }
    )
    dflist += [dg]

df = pd.concat(dflist)
df.to_csv(output_file, index=False)
