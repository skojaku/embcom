"""Generate networks with two communities"""
# %%
import numpy as np
import pandas as pd
from scipy import sparse, stats
import sys

if "snakemake" in sys.modules:
    cave = int(snakemake.params["cave"])
    cdiff = int(snakemake.params["cdiff"])
    n = int(snakemake.params["n"])
    output_file = snakemake.output["output_file"]
else:
    cin, cout = 30, 5
    n = 500
    output_file = "../data/networks/2coms"

#
# Load
#
def generate_dcSBM(cin, cout, N):
    n = int(np.floor(N / 2))
    pin, pout = cin / N, cout / N
    gids = np.concatenate([np.zeros(n), np.ones(n)])
    # d = (cin + cout) / 2

    within_edges = set([])
    target_num = stats.binom.rvs(int(n * (n - 1) / 2 * 2), pin, size=1)[0]
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
    target_num = stats.binom.rvs(n * n, pout, size=1)[0]
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
cin = (cave + cdiff) / 2
cout = cave - cin
A = generate_dcSBM(cin, cout, n)

# %%
# Save
#
sparse.save_npz(output_file, A)
