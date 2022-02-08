import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.append(os.path.abspath(os.path.join("./libs/lfr_benchmark")))
from lfr_benchmark.generator import NetworkGenerator as NetworkGenerator

if "snakemake" in sys.modules:
    N = float(snakemake.params["n"])
    k = float(snakemake.params["k"])
    tau = float(snakemake.params["tau"])
    tau2 = float(snakemake.params["tau2"])
    mu = float(snakemake.params["mu"])
    output_net_file = snakemake.output["output_net"]
    output_community_file = snakemake.output["output_community_file"]

    maxk = snakemake.params["maxk"]
    maxc = snakemake.params["maxc"]

    if (maxk is None) or (maxk == "None"):
        maxk = int(np.sqrt(10 * N))
    else:
        maxk = int(maxk)

    if (maxc is None) or (maxc == "None"):
        maxc = int(np.ceil(N / 4))
        # maxc = int(np.ceil(N/10))
    else:
        maxc = int(maxc)

    minc = 100

else:
    input_file = "../data/"
    output_file = "../data/"

params = {
    "N": N,
    "k": k,
    "maxk": maxk,
    "minc": minc,
    "maxc": maxc,
    "tau": tau,
    "tau2": tau2,
}


root = Path().parent.absolute()
ng = NetworkGenerator()
data = ng.generate(params, mu)
os.chdir(root)

# Load the network
net = data["net"]
community_table = data["community_table"]
params = data["params"]
seed = data["seed"]

community_ids = community_table.sort_values(by="node_id")["community_id"].values.astype(
    int
)
community_ids -= 1  # because the offset is one

# Save
sparse.save_npz(output_net_file, net)
np.savez(output_community_file, group_ids=community_ids)
