# %%
import sys

import infomap
import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
else:
    netfile = "../../data/multi_partition_model/networks/net_n~10000_K~2_cave~20_mu~0.85_sample~2.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~10000_K~2_cave~20_mu~0.85_sample~2.npz"
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
    return cids.astype(int)


group_ids = detect_by_infomap(A, K)
group_ids -= 1

# %%

# %%
# Save
#
np.savez(output_file, group_ids=group_ids)

# %%
