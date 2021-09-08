# %%
import sys

import numpy as np
from scipy import sparse

if "snakemake" in sys.modules:
    n = int(snakemake.params["n"])
    nc = int(snakemake.params["nc"])
    output_file = snakemake.output["output_file"]
else:
    n = 50
    nc = 10
    output_file = "../data/networks/ring_of_cliques"

#
# Construt the network
#

# Generate a membership matrix
K = int(np.round(n / nc))
group_ids = np.kron(np.arange(K), np.ones(nc))
U = sparse.csr_matrix(
    (np.ones_like(group_ids), (np.arange(len(group_ids)), group_ids)), shape=(n, K)
)

# Generate the adjacency matrix of the K independent cliques
A = U @ U.T

# Edges that bridge the cliques
r = np.arange(nc - 1, n - nc, nc)
c = r + 1
r = np.concatenate([r, np.array([n - 1])])
c = np.concatenate([c, np.array([0])])
B = sparse.csr_matrix((np.ones_like(r), (r, c)), shape=(n, n))
A = A + B + B.T

# Remove self-edges
A.setdiag(0)

#
# Save
#
sparse.save_npz(output_file, A)
