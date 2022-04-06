"""Investigate the discrepancy between the actual and analytical spectrum density"""
# %%
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import sparse
import utils
from tqdm import tqdm
from scipy import stats
import pandas as pd
import sys


if "snakemake" in sys.modules:
    param_net = snakemake.params["parameters"]
    output_file = snakemake.output["output_file"]
    L = param_net["L"]
    n_samples = param_net["n_samples"]
else:
    param_net = {"Cave": 50, "mixing_rate": 1 - 1 / np.sqrt(50), "N": 1000, "q": 2}
    L = 10  # window size
    n_samples = 10  # Number of samples
    input_file = "../data/"
    output_file = "../data/"

# ==========
# Setting
# ==========
param_net["memberships"] = utils.get_membership(param_net["N"], param_net["q"])

# ======
# Calculate the ensemble average and variances using simulations
# ======
net_list = [utils.generate_network(**param_net)[0] for i in tqdm(range(n_samples))]
R_list = [utils.make_node2vec_matrix(net, L) for net in tqdm(net_list)]
Rlim_list = [utils.make_node2vec_matrix_limit(net, L) for net in tqdm(net_list)]
# %%
# Calculate the first eivenvectors
#

cin, cout = utils.get_cin_cout(**param_net)
results = []
for i in tqdm(range(len(R_list))):
    R, Rlim = R_list[i], Rlim_list[i]
    R[np.isnan(R) | np.isinf(R)] = 0
    z1, v = sparse.linalg.eigs(R, k=1, which="LR")
    z1lim, vlim = sparse.linalg.eigs(Rlim, k=1, which="LR")
    z1, z1lim = np.real(z1[0]), np.real(z1lim[0])
    v, vlim = np.real(v), np.real(vlim)
    v[np.isnan(v) | np.isinf(v)] = 0
    vlim[np.isnan(vlim) | np.isinf(vlim)] = 0
    v, vlim = v.reshape(-1), vlim.reshape(-1)
    y = param_net["memberships"]
    vscore, vlimscore = roc_auc_score(y, v), roc_auc_score(y, vlim)
    results.append(
        {
            "pearson": np.abs(stats.pearsonr(v, vlim)[0]),
            "score_node2vec": np.abs(vscore - 0.5) + 0.5,
            "score_node2vec_lim": np.abs(vlimscore - 0.5) + 0.5,
        }
    )
df = pd.DataFrame(results)

df.to_csv(output_file, index=False)
# %%
df
