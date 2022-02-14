"""Generate networks with multiple communities."""
# %%
import sys

import graph_tool.all as gt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

if "snakemake" in sys.modules:
    params = snakemake.params["parameters"]
    n = int(params["n"])
    K = int(params["K"])
    cave = int(params["cave"])
    mu = float(params["mu"])
    output_file = snakemake.output["output_file"]
    output_node_file = snakemake.output["output_node_file"]
else:
    n = 600
    K = 25
    cave = 10
    mu = 0.68
    output_file = "../data/networks/multi-coms"


def generate_network(Cave, mixing_rate, N, q):

    memberships = np.sort(np.arange(N) % q)

    q = int(np.max(memberships) + 1)
    N = len(memberships)
    U = sparse.csr_matrix((np.ones(N), (np.arange(N), memberships)), shape=(N, q))

    Cout = np.maximum(1, mixing_rate * Cave)
    Cin = q * Cave - (q - 1) * Cout
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


#
# Load
#
A, memberships = generate_network(cave, mu, n, K)


# %%
# Save
#
sparse.save_npz(output_file, A)
node_ids = np.arange(A.shape[0]).astype(int)
pd.DataFrame({"node_id": node_ids, "membership": memberships}).to_csv(
    output_node_file, index=False
)
