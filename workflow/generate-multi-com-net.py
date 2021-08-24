"""Generate networks with multiple communities."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    cin = int(snakemake.params["cin"])
    cout = int(snakemake.params["cout"])
    n = int(snakemake.params["n"])
    nc = int(snakemake.params["nc"])
    output_file = snakemake.output["output_file"]
else:
    cin, cout = 30, 5
    n = 500
    nc = 500
    output_file = "../data/networks/multi-coms"

# %%
# Load
#
def generate_dcSBM(cin, cout, Nc, N):

    pin, pout = cin / N, cout / N
    K = int(np.floor(N / Nc))
    gids = np.kron(np.arange(K), np.ones(Nc))

    num_within_node_pairs = int(K * Nc * (Nc - 1) / 2)
    num_between_node_pairs = int(N * (N - 1) / 2 - num_within_node_pairs)

    within_edges = set([])
    target_num = stats.binom.rvs(num_within_node_pairs, pin, size=1)[0]
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
    target_num = stats.binom.rvs(num_between_node_pairs, pout, size=1)[0]
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
A = generate_dcSBM(cin, cout, nc, n)


# %%
# Save
#
sparse.save_npz(output_file, A)
