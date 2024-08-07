"""Generate networks with two communities."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    cave = int(snakemake.params["cave"])
    cdiff = int(snakemake.params["cdiff"])
    n = int(snakemake.params["n"])
    output_file = snakemake.output["output_file"]
else:
    cave, cdiff = 100, 20
    n = 500
    output_file = "../data/networks/2coms"

#
# Load
#
def sampling_num_edges(n, p):
    try:
        return stats.binom.rvs(n=n, p=p, size=1)[0]
    except ValueError:
        if n < 100000:
            return np.sum(np.random.rand(n) < p)
        else:
            return stats.poisson.rvs(mu=n * p, size=1)[0]


def generate_dcSBM(cin, cout, N):
    n = int(np.floor(N / 2))
    pin, pout = cin / N, cout / N
    gids = np.concatenate([np.zeros(n), np.ones(n)])
    # d = (cin + cout) / 2

    within_edges = set([])
    target_num = sampling_num_edges(n=int(n * (n - 1) / 2 * 2), p=pin)
    esize = 0
    while esize < target_num:
        r = np.random.choice(2 * n, target_num - esize)
        c = np.random.choice(2 * n, target_num - esize)
        s = (gids[r] == gids[c]) * (r != c)
        r, c = r[s], c[s]
        r, c = np.maximum(r, c), np.minimum(r, c)
        eids = set(r + c * N)
        within_edges = within_edges.union(eids)
        esize = len(within_edges)

    between_edges = set([])
    target_num = sampling_num_edges(n=n * n, p=pout)
    esize = 0
    while esize < target_num:
        r = np.random.choice(2 * n, target_num - esize)
        c = np.random.choice(2 * n, target_num - esize)
        s = (gids[r] != gids[c]) * (r != c)
        r, c = r[s], c[s]
        r, c = np.maximum(r, c), np.minimum(r, c)
        eids = set(r + c * N)
        between_edges = between_edges.union(eids)
        esize = len(between_edges)

    edges = np.array(list(between_edges) + list(within_edges))
    r, c = divmod(edges, N)
    A = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(N, N))
    A = A + A.T
    A.data = np.ones_like(A.data)

    return A


#
# Preprocess
#
nc = n / 2
cin = (n - nc) / n * cdiff + cave
cout = cave - nc / n * cdiff

A = generate_dcSBM(cin, cout, n)

# %%
# Save
#
sparse.save_npz(output_file, A)
