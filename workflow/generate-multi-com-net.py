"""Generate networks with multiple communities."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    cave = int(snakemake.params["cave"])
    cdiff = int(snakemake.params["cdiff"])
    n = int(snakemake.params["n"])
    K = int(snakemake.params["K"])
    output_file = snakemake.output["output_file"]
else:
    K = 2
    cave, cdiff = 100, 40
    n = 200
    nc = 100
    output_file = "../data/networks/multi-coms"

# %%
# Load
#
def sampling_num_edges(n, p):
    if isclose(p, 0):
        return 0
    if isclose(p, 1):
        return n
    try:
        return stats.binom.rvs(n=int(n), p=p, size=1)[0]
    except ValueError:
        if n < 100000:
            return np.sum(np.random.rand(int(n) < p))
        else:
            return stats.poisson.rvs(mu=n * p, size=1)[0]


def generate_dcSBM(cin, cout, Nc, N):

    pin, pout = cin / N, cout / N
    K = int(np.floor(N / Nc))
    gids = np.kron(np.arange(K), np.ones(Nc))

    num_within_node_pairs = int(K * Nc * (Nc - 1) / 2)
    num_between_node_pairs = int(N * (N - 1) / 2 - num_within_node_pairs)

    within_edges = set([])
    target_num = sampling_num_edges(n=num_within_node_pairs, p=pin)
    esize = 0
    while esize < target_num:
        r = np.random.choice(N, target_num - esize)
        c = np.random.choice(N, target_num - esize)
        s = (gids[r] == gids[c]) * (r != c)
        r, c = r[s], c[s]
        r, c = np.maximum(r, c), np.minimum(r, c)
        eids = set(r + c * N)
        within_edges = within_edges.union(eids)
        esize = len(within_edges)

    between_edges = set([])
    target_num = sampling_num_edges(n=num_between_node_pairs, p=pout)
    esize = 0
    while esize < target_num:
        r = np.random.choice(N, target_num - esize)
        c = np.random.choice(N, target_num - esize)
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
nc = int(n / K)
cin = (n - nc) / n * cdiff + cave
cout = cave - nc / n * cdiff
# %%
A = generate_dcSBM(cin, cout, nc, n)


# %%
# Save
#
sparse.save_npz(output_file, A)
