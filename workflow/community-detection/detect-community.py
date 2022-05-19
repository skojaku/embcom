"""Community detection algorithms.

- "infomap": Infomap
- "flatsbm": Degree-corrected SBM
- "nestedsbm": Degree-corrected SBM with nested structure (not implemented)
"""
# %%
import sys

import graph_tool.all as gt
import infomap
import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    model_name = snakemake.params["model_name"]
    output_file = snakemake.output["output_file"]
else:
    netfile = "../../data/multi_partition_model/networks/net_n~2500_K~2_cave~20_mu~0.85_sample~2.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~2500_K~2_cave~20_mu~0.85_sample~2.npz"
    model_name = "flatsbm"
    output_file = "../data/"

# %%
# Load
#
A = sparse.load_npz(netfile)

memberships = pd.read_csv(com_file)["membership"].values.astype(int)
K = len(set(memberships))

# %%
#
# Communiyt detection
#
def detect_by_infomap(A, K):
    r, c, v = sparse.find(A + A.T)
    im = infomap.Infomap("--two-level --directed")
    for i in range(len(r)):
        im.add_link(r[i], c[i], 1)
    im.run()
    cids = np.zeros(A.shape[0])
    for node in im.tree:
        if node.is_leaf:
            cids[node.node_id] = node.module_id
    return np.unique(cids, return_inverse=True)[1]


def detect_by_flatsbm(A, K):
    r, c, v = sparse.find(A)
    g = gt.Graph(directed=False)
    g.add_edge_list(np.vstack([r, c]).T)
    state = gt.minimize_blockmodel_dl(
        g,
        state_args={"B_min": K, "B_max": K},
        multilevel_mcmc_args={"B_max": K, "B_min": K},
    )
    b = state.get_blocks()
    return np.unique(np.array(b.a), return_inverse=True)[1]


if model_name == "infomap":
    group_ids = detect_by_infomap(A, K)
elif model_name == "flatsbm":
    group_ids = detect_by_flatsbm(A, K)

# %%
# Save
#
np.savez(output_file, group_ids=group_ids)
