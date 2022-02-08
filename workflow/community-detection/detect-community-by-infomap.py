# %%
import sys

import infomap
import numpy as np
from scipy import sparse

if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
else:
    netfile = (
        "../data/networks/multi_coms/net_n=1000_K=5_cave=50_cdiff=640_sample=0.npz"
    )
    output_file = "../data/"

# %%
# Load
#
A = sparse.load_npz(netfile)


# %%
#
# Communiyt detection
#
def detect_by_infomap(A):
    r, c, v = sparse.find(A)
    im = infomap.Infomap("--two-level --directed")
    for i in range(len(r)):
        im.add_link(r[i], c[i], v[i])
    im.run()
    cids = np.zeros(A.shape[0])
    for node in im.tree:
        if node.is_leaf:
            cids[node.node_id] = node.module_id
    return cids.astype(int)


group_ids = detect_by_infomap(A)
group_ids -= 1

#
# Save
#
np.savez(output_file, group_ids=group_ids)
