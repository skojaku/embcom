import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace


configfile: "workflow/config.yaml"


# =================
# Utility function
# =================
def make_filename(prefix, ext, names):
    retval = prefix
    for key in names:
        retval += "_" + str(key) + "={" + str(key) + "}"
    return retval + "." + ext


def to_paramspace(dict_list):
    if isinstance(dict_list, list) is False:
        dict_list = [dict_list]
    my_dict = {}
    cols = []
    for dic in dict_list:
        my_dict.update(dic)
        cols += list(dic.keys())
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    df = pd.DataFrame(permutations_dicts)
    df = df[cols]
    return Paramspace(df, filename_params="*")


# =========
# Directory
# =========

DATA_DIR = config["data_dir"]
FIG_DIR = "figs"
RES_DIR = j(DATA_DIR, "results")

PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

NET_DIR = j(DATA_DIR, "multi_partition_model", "networks")
EMB_DIR = j(DATA_DIR, "multi_partition_model", "embedding")
COM_DIR = j(DATA_DIR, "multi_partition_model", "communities")
EVA_DIR = j(DATA_DIR, "multi_partition_model", "evaluations")
VAL_SPEC_DIR = j(DATA_DIR, "multi_partition_model", "spectral_analysis")

# =========
# FIGURES
# =========

# ================================
# Networks and communities
# ================================

net_params = {
    "n": [2500, 5000, 10000, 50000, 100000],  # Network size
    # "n": [10000, 100000], # Network size
    "K": [2],  # Number of communities
    #"K": [2, 16, 32, 64, 100, 200, 500, 1000],  # Number of communities
    # "K": [2, 16, 32], # Number of communities
    "cave": [10, 20, 50],  # average degree
    # "cave": [10, 20, 50], # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    # "mu": [0.05, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95,0.68, 0.85], # detectbility threshold
    "sample": np.arange(3),  # Number of samples
    # "sample": np.arange(10), # Number of samples
}

# Convert to a paramspace
net_paramspace = to_paramspace(net_params)
NET_FILE = j(NET_DIR, f"net_{net_paramspace.wildcard_pattern}.npz")
NODE_FILE = j(NET_DIR, f"node_{net_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================
emb_params = {
    "model_name": [
        "node2vec",
        "leigenmap",
        "modspec",
        "levy-word2vec",
        "linearized-node2vec",
        "non-backtrack-node2vec",
    ],
    # "model_name": ["node2vec", "glove", "depthfirst-node2vec"],
    # "model_name": ["leigenmap", "modspec", "nonbacktracking"],
    "window_length": [10],
    "dim": [0, 64],
}

emb_paramspace = to_paramspace([net_params, emb_params])

EMB_FILE = j(EMB_DIR, f"{emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
com_detect_params = {
    "model_name": ["infomap"],
}
com_detect_paramspace = to_paramspace([net_params, com_detect_params])

# Community detection
COM_DETECT_FILE = j(COM_DIR, f"{com_detect_paramspace.wildcard_pattern}.npz")

# Community detection by clustering to embedding
clustering_params = {
    "metric": ["cosine", "euclidean"],
    "clustering": ["voronoi", "kmeans"],
}
com_detect_emb_paramspace = to_paramspace([net_params, emb_params, clustering_params])
COM_DETECT_EMB_FILE = j(
    COM_DIR, f"clus_{com_detect_emb_paramspace.wildcard_pattern}.npz"
)


# ==========
# Evaluation
# ==========
eval_params = {
    "scoreType": ["esim", "nmi"],
}
eva_emb_paramspace = to_paramspace([eval_params, net_params, emb_params, clustering_params])
EVAL_EMB_FILE = j(EVA_DIR, f"score_clus_{eva_emb_paramspace.wildcard_pattern}.npz")

eva_paramspace = to_paramspace([eval_params, net_params, com_detect_params])
EVAL_FILE = j(EVA_DIR, f"score_{eva_paramspace.wildcard_pattern}.npz")

EVAL_CONCAT_FILE = j(EVA_DIR, f"esim-all-result.csv")


# ===============================
# Validating detectability limit
# ===============================
bipartition_params = {
    "Cave": [10, 20, 50],
    "mixing_rate": [0.5],
    "N": [1000, 5000],
    "q": [2],
    "matrixType": ["node2vec", "linearized-node2vec"],
    "L": [1, 10, 50],
    "n_samples": [10],
}

bipartition_paramspace = to_paramspace([bipartition_params])
SPECTRAL_DENSITY_FILE = j(
    VAL_SPEC_DIR, f"{bipartition_paramspace.wildcard_pattern}.csv"
)

# =====
# Figures
# =====
FIG_SPECTRAL_DENSITY_FILE = j(FIG_DIR, "spectral-density", f"{bipartition_paramspace.wildcard_pattern}.pdf")

# ======
# RULES
# ======


rule all:
    input:
        #expand(SPECTRAL_DENSITY_FILE, **bipartition_params), #expand(EVAL_FILE, **net_params, **com_detect_params),
        expand(EVAL_EMB_FILE, **net_params, **emb_params, **clustering_params, **eval_params),
        #expand(COM_DETECT_FILE, **net_params, **com_detect_params),
        expand(COM_DETECT_EMB_FILE, **net_params, **emb_params, **clustering_params)

rule figs:
    input:
        expand(FIG_SPECTRAL_DENSITY_FILE, **bipartition_params)

#
# network generation
#
rule generate_net_multi_partition_model:
    params:
        parameters=net_paramspace.instance,
    output:
        output_file=NET_FILE,
        output_node_file=NODE_FILE,
    script:
        "workflow/net_generator/generate-net-by-multi-partition-model.py"


#
# Embedding
#
rule embedding_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EMB_FILE,
    params:
        parameters=emb_paramspace.instance,
    script:
        "workflow/embedding/embedding.py"


#
# Clustering
#
rule voronoi_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="voronoi",
    script:
        "workflow/community-detection/voronoi-clustering.py"


rule kmeans_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="kmeans",
    script:
        "workflow/community-detection/kmeans-clustering.py"


rule community_detection_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_FILE,
    params:
        parameters=com_detect_paramspace.instance,
    script:
        "workflow/community-detection/detect-community-by-infomap.py"


#
# Evaluation
#
rule evaluate_communities:
    input:
        detected_group_file=COM_DETECT_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_FILE,
    params:
        parameters=eva_paramspace.instance,
    script:
        "workflow/evaluation/eval-com-detect-score.py"

rule evaluate_communities_for_embedding:
    input:
        detected_group_file=COM_DETECT_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_EMB_FILE,
    params:
        parameters=eva_paramspace.instance,
    script:
        "workflow/evaluation/eval-com-detect-score.py"

rule concatenate_results:
    input:
        input_dir = EVA_DIR
    output:
        output_file=EVAL_CONCAT_FILE
    script:
        "workflow/evaluation/concatenate_results.py"

#
# Validating the detectability condition
#
rule calc_spectral_density_linearized_node2vec:
    output:
        output_file=SPECTRAL_DENSITY_FILE,
    params:
        parameters=bipartition_paramspace.instance,
    script:
        "workflow/spectral-density-analysis/calc-spec-density-node2vec.py"


#
# Plot
#
rule plot_spectral_density:
    input:
        input_file=SPECTRAL_DENSITY_FILE,
    output:
        output_file=FIG_SPECTRAL_DENSITY_FILE
    script:
        "workflow/plot/plot-spectral-density.py"
