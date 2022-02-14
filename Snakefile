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
        retval += "_"+str(key)+"={"+ str(key) + "}"
    return retval + "." + ext

def to_paramspace(dict_list):
    if isinstance(dict_list, list) is False:
        dict_list = [dict_list]
    my_dict = {}
    cols = []
    for dic in dict_list:
        my_dict.update(dic)
        cols+=list(dic.keys())
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    df = pd.DataFrame(permutations_dicts)
    df = df[cols]
    return Paramspace(df, filename_params = "*")

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

# =========
# FIGURES
# =========

# ================================
# Networks and communities
# ================================

net_params = {
    "n": [10000, 100000], # Network size
    "K": [2, 16, 32], # Number of communities
    "cave": [10, 20, 50], # average degree
    "mu": [0.1, 0.3, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 0.68, 0.85], # detectbility threshold
    #"mu": [0.05, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95,0.68, 0.85], # detectbility threshold
    "sample": np.arange(10), # Number of samples
}

# Convert to a paramspace
net_paramspace = to_paramspace(net_params)
NET_FILE = j(NET_DIR, f"net_{net_paramspace.wildcard_pattern}.npz" )
NODE_FILE = j(NET_DIR, f"node_{net_paramspace.wildcard_pattern}.npz" )

# =================
# Embedding
# =================
emb_params = {
    "model_name": ["node2vec", "leigenmap", "modspec", "levy-word2vec"],
    #"model_name": ["node2vec", "glove", "depthfirst-node2vec"],
    #"model_name": ["leigenmap", "modspec", "nonbacktracking"],
    "window_length": [10],
    "dim": [64],
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

# Community detection by voronoi clustering to embedding
com_detect_emb_paramspace = to_paramspace([net_params, emb_params])
COM_DETECT_EMB_FILE = j(COM_DIR, f"vonoroi_clus_{com_detect_emb_paramspace.wildcard_pattern}.npz")


# ==========
# Evaluation
# ==========

eva_emb_paramspace = to_paramspace([net_params, emb_params])
EVAL_ESIM_EMB_FILE = j(EVA_DIR, f"esim_voronoi_clus_{eva_emb_paramspace.wildcard_pattern}.npz")

eva_paramspace = to_paramspace([net_params, com_detect_params])
EVAL_ESIM_FILE = j(EVA_DIR, f"esim_{eva_paramspace.wildcard_pattern}.npz")

# ======
# RULES
# ======

rule all:
    input:
        expand(EVAL_ESIM_FILE, **net_params, **com_detect_params),
        expand(EVAL_ESIM_EMB_FILE, **net_params, **emb_params),
        #expand(EMB_FILE, **net_params, **emb_params),
        #expand(COM_DETECT_FILE, **net_params, **com_detect_params),
        #expand(COM_DETECT_EMB_FILE, **net_params, **emb_params)

rule generate_net_multi_partition_model:
    params:
        parameters = net_paramspace.instance
    output:
        output_file=NET_FILE,
        output_node_file=NODE_FILE,
    script:
        "workflow/net_generator/generate-net-by-multi-partition-model.py"

rule embedding_multi_partition_model:
    input:
        net_file=NET_FILE,
    output:
        output_file=EMB_FILE,
    params:
        parameters = emb_paramspace.instance
    script:
        "workflow/embedding/embedding.py"


rule voronoi_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters = com_detect_emb_paramspace.instance
    script:
        "workflow/community-detection/voronoi-clustering.py"

rule community_detection_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file= COM_DETECT_FILE,
    params:
        parameters = com_detect_paramspace.instance
    script:
        "workflow/community-detection/detect-community-by-infomap.py"

rule evaluate_communities_by_esim:
    input:
        detected_group_file = COM_DETECT_FILE,
        com_file = NODE_FILE
    output:
        output_file = EVAL_ESIM_FILE
    script:
        "workflow/evaluation/eval-esim.py"

rule evaluate_communities_by_esim_for_embedding:
    input:
        detected_group_file = COM_DETECT_EMB_FILE,
        com_file = NODE_FILE
    output:
        output_file = EVAL_ESIM_EMB_FILE
    script:
        "workflow/evaluation/eval-esim.py"

#        netfile=SIM_MULTI_FIXED_SZ_COM_NET,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_FILE,
#    script:

#rule multi_fixed_size_com_embedding:
#    input:
#        netfile=SIM_MULTI_FIXED_SZ_COM_NET,
#    output:
#        embfile=MULTI_FIXED_SZ_COM_EMB_FILE,
#    params:
#        model_name=lambda wildcards: wildcards.model_name,
#        dim=lambda wildcards: wildcards.dim,
#        window_length=lambda wildcards: wildcards.window_length,
#        directed="undirected",
#        num_walks=10,
#    script:
#        "workflow/embedding.py"
#
#
#rule eval_auc_fixed_size_com_embedding:
#    input:
#        emb_file=MULTI_FIXED_SZ_COM_EMB_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    output:
#        output_file=MULTI_FIXED_SZ_COM_AUC_FILE,
#        output_sim_file=MULTI_FIXED_SZ_COM_SIM_FILE,
#    script:
#        "workflow/eval-community.py"
#
#
#rule eval_fixed_size_com_embedding_kmeans:
#    input:
#        emb_file=MULTI_FIXED_SZ_COM_EMB_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    output:
#        output_sim_file=MULTI_FIXED_SZ_COM_KMEANS_FILE,
#    script:
#        "workflow/eval-community-kmeans.py"
#
#
#rule concat_auc_result_fixed_size_file:
#    input:
#        input_files=MULTI_FIXED_SZ_COM_AUC_FILE_ALL,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_AUC_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_kmeans_result_fixed_size_file:
#    input:
#        input_files=MULTI_FIXED_SZ_COM_KMEANS_FILE_ALL,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_KMEANS_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_community_detection_result_fixed_size_file:
#    input:
#        input_files=MULTI_FIXED_SZ_COM_COM_DETECT_FILE_ALL,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_COM_DETECT_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_dist_result_fixed_size_file:
#    input:
#        input_files=MULTI_FIXED_SZ_COM_DIST_FILE_ALL,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_DIST_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_separatability_result_fixed_size_file:
#    input:
#        input_files=MULTI_FIXED_SZ_COM_SEPARATABILITY_FILE_ALL,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_SEPARATABILITY_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule detect_fixed_size_community_by_infomap:
#    input:
#        netfile=SIM_MULTI_FIXED_SZ_COM_NET,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_FILE,
#    script:
#        "workflow/detect-community-by-infomap.py"
#
#
#rule eval_fixed_size_detected_community:
#    input:
#        com_file=MULTI_FIXED_SZ_COM_FILE,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_COM_DETECT_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    script:
#        "workflow/eval-detected-community.py"
#
#
#rule eval_multi_fixed_sz_com_distances:
#    input:
#        emb_file=MULTI_FIXED_SZ_COM_EMB_FILE,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_DIST_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#        num_samples=10000,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/eval-community-distances.py"
#
#
#rule eval_multi_fixed_sz_com_separatability:
#    input:
#        emb_file=MULTI_FIXED_SZ_COM_EMB_FILE,
#    output:
#        output_file=MULTI_FIXED_SZ_COM_SEPARATABILITY_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/eval-community-separatability.py"
#
#
## ==================================
## Multiple variable-size communities
## ==================================
#MULTI_COM_NET_DIR = j(DATA_DIR, "networks", "multi_coms")
#MULTI_COM_EMB_DIR = j(DATA_DIR, "embeddings", "multi_coms")
#MULTI_COM_DIR = j(DATA_DIR, "communities", "multi_coms")
#sim_net_params = {
#    "n": [1000, 10000, 100000, 1000000],
#    #"n": [1000, 10000, 100000, 1000000],
#    "cave": [10, 50],
#    "cdiff": [100, 300],
#    "K": [2, 25],
#    "sample": np.arange(10),
#}
#SIM_MULTI_COM_NET = j(
#    MULTI_COM_NET_DIR, "net_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}.npz"
#)
#SIM_MULTI_COM_NET_ALL = expand(SIM_MULTI_COM_NET, **sim_net_params)
#
#emb_params_rw = {  # parameter for methods baesd on random walks
#    "model_name": ["node2vec"],
#    #"model_name": ["node2vec", "glove", "depthfirst-node2vec"],
#    "window_length": [10],
#    "dim": [10, 24, 64],
#    #"dim": [1, 64] + [k - 1 for k in sim_net_params["K"]],
#}
#emb_params = {
#        #"model_name": ["leigenmap", "modspec"],
#    "model_name": ["leigenmap", "modspec"],
#    "window_length": [10],
#    "dim": [1, 10, 24, 64],
#    #"dim": [1, 10, 24, 64],
#    #"dim": [1, 64] + [k - 1 for k in sim_net_params["K"]],
#}
#MULTI_COM_EMB_FILE = j(
#    MULTI_COM_EMB_DIR,
#    "embnet_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
#)
#MULTI_COM_EMB_FILE_ALL = expand(
#    MULTI_COM_EMB_FILE, **sim_net_params, **emb_params
#) + expand(MULTI_COM_EMB_FILE, **sim_net_params, **emb_params_rw)
#
## Community detection
#MULTI_COM_FILE = j(
#    MULTI_COM_DIR,
#    "community_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}.npz",
#)
#com_detect_params = {
#    "model_name": ["infomap"],
#}
#MULTI_COM_FILE_ALL = expand(MULTI_COM_FILE, **sim_net_params, **com_detect_params)
#
#
## Derived
#MULTI_COM_AUC_FILE = j(
#    RES_DIR,
#    "multi_coms",
#    "auc",
#    "auc_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#MULTI_COM_SIM_FILE = j(
#    RES_DIR,
#    "multi_coms",
#    "similarity",
#    "similarity_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#MULTI_COM_KMEANS_FILE = j(
#    RES_DIR,
#    "multi_coms",
#    "kmeans",
#    "kmeans_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#MULTI_COM_COM_DETECT_FILE = j(
#    RES_DIR,
#    "multi_coms",
#    "community_detection",
#    "result_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}.csv",
#)
#MULTI_COM_DIST_FILE = j(
#    RES_DIR,
#    "multi_coms",
#    "distances",
#    "result_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#MULTI_COM_SEPARATABILITY_FILE = j(
#    RES_DIR,
#    "multi_coms",
#    "separatability",
#    "result_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#MULTI_COM_AUC_FILE_ALL = expand(
#    MULTI_COM_AUC_FILE, **sim_net_params, **emb_params
#) + expand(MULTI_COM_AUC_FILE, **sim_net_params, **emb_params_rw)
#MULTI_COM_SIM_FILE_ALL = expand(
#    MULTI_COM_SIM_FILE, **sim_net_params, **emb_params
#) + expand(MULTI_COM_SIM_FILE, **sim_net_params, **emb_params_rw)
#MULTI_COM_KMEANS_FILE_ALL = expand(
#    MULTI_COM_KMEANS_FILE, **sim_net_params, **emb_params
#) + expand(MULTI_COM_KMEANS_FILE, **sim_net_params, **emb_params_rw)
#MULTI_COM_COM_DETECT_FILE_ALL = expand(
#    MULTI_COM_COM_DETECT_FILE, **sim_net_params, **com_detect_params
#)
#MULTI_COM_DIST_FILE_ALL = expand(
#    MULTI_COM_DIST_FILE, **sim_net_params, **emb_params
#) + expand(MULTI_COM_DIST_FILE, **sim_net_params, **emb_params_rw)
#MULTI_COM_SEPARATABILITY_FILE_ALL = expand(
#    MULTI_COM_SEPARATABILITY_FILE, **sim_net_params, **emb_params
#) + expand(MULTI_COM_SEPARATABILITY_FILE, **sim_net_params, **emb_params_rw)
#
#
#MULTI_COM_AUC_RES_FILE = j(RES_DIR, "multi_coms", "results", "auc.csv")
#MULTI_COM_KMEANS_RES_FILE = j(RES_DIR, "multi_coms", "results", "kmeans.csv")
#MULTI_COM_COM_DETECT_RES_FILE = j(
#    RES_DIR, "multi_coms", "results", "community_detection.csv"
#)
#MULTI_COM_DIST_RES_FILE = j(RES_DIR, "multi_coms", "results", "distances.csv")
#MULTI_COM_SEPARATABILITY_RES_FILE = j(
#    RES_DIR, "multi_coms", "results", "separatability.csv"
#)
#
#
#rule generate_multi_com_net:
#    params:
#        cave=lambda wildcards: int(wildcards.cave),
#        cdiff=lambda wildcards: float(wildcards.cdiff),
#        n=lambda wildcards: int(wildcards.n),
#        K=lambda wildcards: int(wildcards.K),
#    output:
#        output_file=SIM_MULTI_COM_NET,
#    script:
#        "workflow/generate-net-by-multi-partition-model.py"
#
#
#rule multi_com_embedding:
#    input:
#        netfile=SIM_MULTI_COM_NET,
#    output:
#        embfile=MULTI_COM_EMB_FILE,
#    params:
#        model_name=lambda wildcards: wildcards.model_name,
#        dim=lambda wildcards: wildcards.dim,
#        window_length=lambda wildcards: wildcards.window_length,
#        directed="undirected",
#        num_walks=10,
#    script:
#        "workflow/embedding.py"
#
#
#rule eval_auc_multi_com_embedding:
#    input:
#        emb_file=MULTI_COM_EMB_FILE,
#    params:
#        K=lambda wildcards: wildcards.K,
#    output:
#        output_file=MULTI_COM_AUC_FILE,
#        output_sim_file=MULTI_COM_SIM_FILE,
#    script:
#        "workflow/eval-community.py"
#
#
#rule eval_multi_com_embedding_kmeans:
#    input:
#        emb_file=MULTI_COM_EMB_FILE,
#    params:
#        K=lambda wildcards: wildcards.K,
#    output:
#        output_sim_file=MULTI_COM_KMEANS_FILE,
#    script:
#        "workflow/eval-community-kmeans.py"
#
#
#rule concat_auc_result_multi_com_file:
#    input:
#        input_files=MULTI_COM_AUC_FILE_ALL,
#    output:
#        output_file=MULTI_COM_AUC_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_kmeans_result_multi_com_file:
#    input:
#        input_files=MULTI_COM_KMEANS_FILE_ALL,
#    output:
#        output_file=MULTI_COM_KMEANS_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_community_detection_result_multi_com_file:
#    input:
#        input_files=MULTI_COM_COM_DETECT_FILE_ALL,
#    output:
#        output_file=MULTI_COM_COM_DETECT_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_dist_result_multi_com_file:
#    input:
#        input_files=MULTI_COM_DIST_FILE_ALL,
#    output:
#        output_file=MULTI_COM_DIST_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_separatability_result_multi_com_file:
#    input:
#        input_files=MULTI_COM_SEPARATABILITY_FILE_ALL,
#    output:
#        output_file=MULTI_COM_SEPARATABILITY_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule detect_community_by_infomap:
#    input:
#        netfile=SIM_MULTI_COM_NET,
#    output:
#        output_file=MULTI_COM_FILE,
#    script:
#        "workflow/detect-community-by-infomap.py"
#
#
#rule eval_detected_community:
#    input:
#        com_file=MULTI_COM_FILE,
#    output:
#        output_file=MULTI_COM_COM_DETECT_FILE,
#    params:
#        K=lambda wildcards: wildcards.K,
#    script:
#        "workflow/eval-detected-community.py"
#
#
#rule eval_multi_com_distances:
#    input:
#        emb_file=MULTI_COM_EMB_FILE,
#    output:
#        output_file=MULTI_COM_DIST_FILE,
#    params:
#        K=lambda wildcards: wildcards.K,
#        num_samples=10000,
#    script:
#        "workflow/eval-community-distances.py"
#
#
#rule eval_multi_com_separatability:
#    input:
#        emb_file=MULTI_COM_EMB_FILE,
#    output:
#        output_file=MULTI_COM_SEPARATABILITY_FILE,
#    params:
#        K=lambda wildcards: wildcards.K,
#    script:
#        "workflow/eval-community-separatability.py"
#
#
## ==================================
## Ring of Cliques
## ==================================
#RING_OF_CLIQUE_NET_DIR = j(DATA_DIR, "networks", "ring_of_cliques")
#RING_OF_CLIQUE_EMB_DIR = j(DATA_DIR, "embeddings", "ring_of_cliques")
#RING_OF_CLIQUE_DIR = j(DATA_DIR, "communities", "ring_of_cliques")
#sim_net_params = {
#    "n": [1000, 10000, 100000, 1000000],
#    #"n": [1000, 10000, 100000, 1000000],
#    "cave": [10, 50],
#    "cdiff": [100, 300],
#    "nc": [100],
#    #"nc": [10, 100],
#    "sample": np.arange(10),
#}
#SIM_RING_OF_CLIQUE_NET = j(
#    RING_OF_CLIQUE_NET_DIR,
#    "net_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}.npz",
#)
#SIM_RING_OF_CLIQUE_NET_ALL = expand(SIM_RING_OF_CLIQUE_NET, **sim_net_params)
#
#emb_params_rw = {  # parameter for methods baesd on random walks
#    "model_name": ["node2vec"],
#    #"model_name": ["node2vec", "glove", "depthfirst-node2vec"],
#    "window_length": [10],
#    "dim": [64],
#}
#emb_params = {
#        #"model_name": ["leigenmap", "modspec", "nonbacktracking"],
#    "model_name": ["leigenmap", "modspec"],
#    #"model_name": ["leigenmap", "modspec"],
#    "window_length": [10],
#    "dim": [1, 64],
#}
#RING_OF_CLIQUE_EMB_FILE = j(
#    RING_OF_CLIQUE_EMB_DIR,
#    "embnet_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
#)
#RING_OF_CLIQUE_EMB_FILE_ALL = expand(
#    RING_OF_CLIQUE_EMB_FILE, **sim_net_params, **emb_params
#) + expand(RING_OF_CLIQUE_EMB_FILE, **sim_net_params, **emb_params_rw)
#
## Community detection
#RING_OF_CLIQUE_FILE = j(
#    RING_OF_CLIQUE_DIR,
#    "community_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}.npz",
#)
#com_detect_params = {
#    "model_name": ["infomap"],
#}
#RING_OF_CLIQUE_FILE_ALL = expand(
#    RING_OF_CLIQUE_FILE, **sim_net_params, **com_detect_params
#)
#
#
## Derived
#RING_OF_CLIQUE_AUC_FILE = j(
#    RES_DIR,
#    "ring_of_cliques",
#    "auc",
#    "auc_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#RING_OF_CLIQUE_SIM_FILE = j(
#    RES_DIR,
#    "ring_of_cliques",
#    "similarity",
#    "similarity_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#RING_OF_CLIQUE_KMEANS_FILE = j(
#    RES_DIR,
#    "ring_of_cliques",
#    "kmeans",
#    "kmeans_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#RING_OF_CLIQUE_COM_DETECT_FILE = j(
#    RES_DIR,
#    "ring_of_cliques",
#    "community_detection",
#    "result_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}.csv",
#)
#RING_OF_CLIQUE_DIST_FILE = j(
#    RES_DIR,
#    "ring_of_cliques",
#    "distances",
#    "result_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#RING_OF_CLIQUE_SEPARATABILITY_FILE = j(
#    RES_DIR,
#    "ring_of_cliques",
#    "separatability",
#    "result_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#RING_OF_CLIQUE_AUC_FILE_ALL = expand(
#    RING_OF_CLIQUE_AUC_FILE, **sim_net_params, **emb_params
#) + expand(RING_OF_CLIQUE_AUC_FILE, **sim_net_params, **emb_params_rw)
#RING_OF_CLIQUE_SIM_FILE_ALL = expand(
#    RING_OF_CLIQUE_SIM_FILE, **sim_net_params, **emb_params
#) + expand(RING_OF_CLIQUE_SIM_FILE, **sim_net_params, **emb_params_rw)
#RING_OF_CLIQUE_KMEANS_FILE_ALL = expand(
#    RING_OF_CLIQUE_KMEANS_FILE, **sim_net_params, **emb_params
#) + expand(RING_OF_CLIQUE_KMEANS_FILE, **sim_net_params, **emb_params_rw)
#RING_OF_CLIQUE_COM_DETECT_FILE_ALL = expand(
#    RING_OF_CLIQUE_COM_DETECT_FILE, **sim_net_params, **com_detect_params
#)
#RING_OF_CLIQUE_DIST_FILE_ALL = expand(
#    RING_OF_CLIQUE_DIST_FILE, **sim_net_params, **emb_params
#) + expand(RING_OF_CLIQUE_DIST_FILE, **sim_net_params, **emb_params_rw)
#RING_OF_CLIQUE_SEPARATABILITY_FILE_ALL = expand(
#    RING_OF_CLIQUE_SEPARATABILITY_FILE, **sim_net_params, **emb_params
#) + expand(RING_OF_CLIQUE_SEPARATABILITY_FILE, **sim_net_params, **emb_params_rw)
#
#RING_OF_CLIQUE_AUC_RES_FILE = j(RES_DIR, "ring_of_cliques", "results", "auc.csv")
#RING_OF_CLIQUE_KMEANS_RES_FILE = j(RES_DIR, "ring_of_cliques", "results", "kmeans.csv")
#RING_OF_CLIQUE_COM_DETECT_RES_FILE = j(
#    RES_DIR, "ring_of_cliques", "results", "community_detection.csv"
#)
#RING_OF_CLIQUE_DIST_RES_FILE = j(RES_DIR, "ring_of_cliques", "results", "distances.csv")
#RING_OF_CLIQUE_SEPARATABILITY_RES_FILE = j(
#    RES_DIR, "ring_of_cliques", "results", "separatability.csv"
#)
#
#
#rule generate_ring_of_clique_net:
#    params:
#        n=lambda wildcards: int(wildcards.n),
#        nc=lambda wildcards: int(wildcards.nc),
#    output:
#        output_file=SIM_RING_OF_CLIQUE_NET,
#    script:
#        "workflow/generate-ring-of-cliques.py"
#
#
#rule ring_of_clique_embedding:
#    input:
#        netfile=SIM_RING_OF_CLIQUE_NET,
#    output:
#        embfile=RING_OF_CLIQUE_EMB_FILE,
#    params:
#        model_name=lambda wildcards: wildcards.model_name,
#        dim=lambda wildcards: wildcards.dim,
#        window_length=lambda wildcards: wildcards.window_length,
#        directed="undirected",
#        num_walks=10,
#    script:
#        "workflow/embedding.py"
#
#
#rule eval_auc_ring_of_clique_embedding:
#    input:
#        emb_file=RING_OF_CLIQUE_EMB_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    output:
#        output_file=RING_OF_CLIQUE_AUC_FILE,
#        output_sim_file=RING_OF_CLIQUE_SIM_FILE,
#    script:
#        "workflow/eval-community.py"
#
#
#rule eval_ring_of_clique_embedding_kmeans:
#    input:
#        emb_file=RING_OF_CLIQUE_EMB_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    output:
#        output_sim_file=RING_OF_CLIQUE_KMEANS_FILE,
#    script:
#        "workflow/eval-community-kmeans.py"
#
#
#rule concat_auc_result_ring_of_clique_file:
#    input:
#        input_files=RING_OF_CLIQUE_AUC_FILE_ALL,
#    output:
#        output_file=RING_OF_CLIQUE_AUC_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_kmeans_result_ring_of_clique_file:
#    input:
#        input_files=RING_OF_CLIQUE_KMEANS_FILE_ALL,
#    output:
#        output_file=RING_OF_CLIQUE_KMEANS_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_community_detection_result_ring_of_clique_file:
#    input:
#        input_files=RING_OF_CLIQUE_COM_DETECT_FILE_ALL,
#    output:
#        output_file=RING_OF_CLIQUE_COM_DETECT_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_dist_result_ring_of_clique_file:
#    input:
#        input_files=RING_OF_CLIQUE_DIST_FILE_ALL,
#    output:
#        output_file=RING_OF_CLIQUE_DIST_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_separatability_result_ring_of_clique_file:
#    input:
#        input_files=RING_OF_CLIQUE_SEPARATABILITY_FILE_ALL,
#    output:
#        output_file=RING_OF_CLIQUE_SEPARATABILITY_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule detect_ring_of_clique_community_by_infomap:
#    input:
#        netfile=SIM_RING_OF_CLIQUE_NET,
#    output:
#        output_file=RING_OF_CLIQUE_FILE,
#    script:
#        "workflow/detect-community-by-infomap.py"
#
#
#rule eval_ring_of_clique_detected_community:
#    input:
#        com_file=RING_OF_CLIQUE_FILE,
#    output:
#        output_file=RING_OF_CLIQUE_COM_DETECT_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    script:
#        "workflow/eval-detected-community.py"
#
#
#rule eval_ring_of_clique_distances:
#    input:
#        emb_file=RING_OF_CLIQUE_EMB_FILE,
#    output:
#        output_file=RING_OF_CLIQUE_DIST_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#        num_samples=10000,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/eval-community-distances.py"
#
#
#rule eval_ring_of_clique_separatability:
#    input:
#        emb_file=RING_OF_CLIQUE_EMB_FILE,
#    output:
#        output_file=RING_OF_CLIQUE_SEPARATABILITY_FILE,
#    params:
#        K=lambda wildcards: int(wildcards.n) / int(wildcards.nc),
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/eval-community-separatability.py"
#
#
## ==================================
## LFR
## ==================================
#LFR_NET_DIR = j(DATA_DIR, "networks", "lfr")
#LFR_EMB_DIR = j(DATA_DIR, "embeddings", "lfr")
#LFR_COM_DIR = j(DATA_DIR, "communities", "lfr")
#sim_net_params = {
#    #"n": [1000, 10000, 100000],
#    #"n": [100000],
#    "n": [1000, 10000, 100000, 1000000],
#    "k": [20],
#    #"k": [10],
#    "maxk": [100], # cin - cout
#    "maxc": [None],
#    #"maxc": [250, None],
#    "tau": [2],
#    "tau2": [2,3,4,5,6],
#    "mu": ["0.25", "0.5"],
#    #"mu": ["0.25", "0.5"],
#    #"mu": ["%.2f" % d for d in np.linspace(0.05, 1, 20)],
#    #"sample": np.arange(3),
#    "sample": np.arange(10),
#}
#SIM_LFR_NET = j(
#    LFR_NET_DIR,
#    "net_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}.npz",
#    #"net_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}.npz",
#)
#SIM_LFR_COM_FILE = j(
#    LFR_NET_DIR,
#    "community_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}.npz",
#    #"community_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}.npz",
#)
#com_detect_params = {
#    "model_name": ["infomap"],
#}
#SIM_LFR_NET_ALL = expand(SIM_LFR_NET, **sim_net_params)
#SIM_LFR_COM_ALL = expand(SIM_LFR_COM_FILE, **sim_net_params)
#
#emb_params_rw = {  # parameter for methods baesd on random walks
#    "model_name": ["node2vec"],
#    #"model_name": ["node2vec", "glove", "depthfirst-node2vec"],
#    "window_length": [10],
#    "dim": [1, 64],
#}
#emb_params = {
#    "model_name": ["leigenmap", "modspec"],
#        #"model_name": ["leigenmap", "modspec", "nonbacktracking"],
#    "window_length": [10],
#    "dim": [1, 64],
#}
#LFR_EMB_FILE = j(
#    LFR_EMB_DIR,
#    "embnet_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
#    #"embnet_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
#)
#LFR_EMB_FILE_ALL = expand(LFR_EMB_FILE, **sim_net_params, **emb_params) + expand(
#    LFR_EMB_FILE, **sim_net_params, **emb_params_rw
#)
#
## Community
#LFR_COM_FILE = j(
#    LFR_COM_DIR,
#    "community_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}.npz",
#    #"community_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}.npz",
#)
#
## Derived
#LFR_AUC_FILE = j(
#    RES_DIR,
#    "lfr",
#    "auc",
#    "auc_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#    #"auc_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#LFR_SIM_FILE = j(
#    RES_DIR,
#    "lfr",
#    "similarity",
#    "similarity_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#    #"similarity_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#LFR_KMEANS_FILE = j(
#    RES_DIR,
#    "lfr",
#    "kmeans",
#    "kmeans_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#    #"kmeans_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#LFR_COM_DETECT_FILE = j(
#    RES_DIR,
#    "lfr",
#    "community_detection",
#    "result_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}.csv",
#    #"result_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}.csv",
#)
#LFR_DIST_FILE = j(
#    RES_DIR,
#    "lfr",
#    "distances",
#    "result_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#    #"result_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#LFR_SEPARATABILITY_FILE = j(
#    RES_DIR,
#    "lfr",
#    "separatability",
#    "result_n={n}_k={k}_tau={tau}_tau2={tau2}_mu={mu}_maxk={maxk}_maxc={maxc}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#    #"result_n={n}_k={k}_maxk={maxk}_minc={minc}_maxc={maxc}_tau={tau}_tau2={tau2}_mu={mu}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv",
#)
#LFR_AUC_FILE_ALL = expand(LFR_AUC_FILE, **sim_net_params, **emb_params) + expand(
#    LFR_AUC_FILE, **sim_net_params, **emb_params_rw
#)
#LFR_SIM_FILE_ALL = expand(LFR_SIM_FILE, **sim_net_params, **emb_params) + expand(
#    LFR_SIM_FILE, **sim_net_params, **emb_params_rw
#)
#LFR_KMEANS_FILE_ALL = expand(LFR_KMEANS_FILE, **sim_net_params, **emb_params) + expand(
#    LFR_KMEANS_FILE, **sim_net_params, **emb_params_rw
#)
#LFR_COM_DETECT_FILE_ALL = expand(
#    LFR_COM_DETECT_FILE, **sim_net_params, **com_detect_params
#)
#LFR_DIST_FILE_ALL = expand(LFR_DIST_FILE, **sim_net_params, **emb_params) + expand(
#    LFR_DIST_FILE, **sim_net_params, **emb_params_rw
#)
#LFR_SEPARATABILITY_FILE_ALL = expand(
#    LFR_SEPARATABILITY_FILE, **sim_net_params, **emb_params
#) + expand(LFR_SEPARATABILITY_FILE, **sim_net_params, **emb_params_rw)
#
#LFR_AUC_RES_FILE = j(RES_DIR, "lfr", "results", "auc.csv")
#LFR_KMEANS_RES_FILE = j(RES_DIR, "lfr", "results", "kmeans.csv")
#LFR_COM_DETECT_RES_FILE = j(RES_DIR, "lfr", "results", "community_detection.csv")
#LFR_DIST_RES_FILE = j(RES_DIR, "lfr", "results", "distances.csv")
#LFR_SEPARATABILITY_RES_FILE = j(RES_DIR, "lfr", "results", "separatability.csv")
#
#
#rule generate_lfr_net:
#    params:
#        n=lambda wildcards: int(wildcards.n),
#        k=lambda wildcards: int(wildcards.k),
#        maxk=lambda wildcards: wildcards.maxk,
#        #minc=lambda wildcards: int(wildcards.minc),
#        maxc=lambda wildcards: wildcards.maxc,
#        tau=lambda wildcards: float(wildcards.tau),
#        tau2=lambda wildcards: float(wildcards.tau2),
#        mu=lambda wildcards: float(wildcards.mu),
#    output:
#        output_net=SIM_LFR_NET,
#        output_community_file=SIM_LFR_COM_FILE
#    script:
#        "workflow/generate-lfr-net.py"
#
#
#rule lfr_embedding:
#    input:
#        netfile=SIM_LFR_NET,
#    output:
#        embfile=LFR_EMB_FILE,
#    params:
#        model_name=lambda wildcards: wildcards.model_name,
#        dim=lambda wildcards: wildcards.dim,
#        window_length=lambda wildcards: wildcards.window_length,
#        directed="undirected",
#        num_walks=10,
#    script:
#        "workflow/embedding.py"
#
#
#rule eval_auc_lfr_embedding:
#    input:
#        emb_file=LFR_EMB_FILE,
#        com_file=SIM_LFR_COM_FILE,
#    params:
#        K=1,
#    output:
#        output_file=LFR_AUC_FILE,
#        output_sim_file=LFR_SIM_FILE,
#    script:
#        "workflow/eval-community.py"
#
#rule eval_lfr_embedding_kmeans:
#    input:
#        emb_file=LFR_EMB_FILE,
#        com_file=SIM_LFR_COM_FILE,
#    params:
#        K=1,
#    output:
#        output_sim_file=LFR_KMEANS_FILE,
#    script:
#        "workflow/eval-community-kmeans.py"
#
#rule detect_lfr_community_by_infomap:
#    input:
#        netfile=SIM_LFR_NET,
#    output:
#        output_file=LFR_COM_FILE,
#    script:
#        "workflow/detect-community-by-infomap.py"
#
#
#rule eval_lfr_detected_community:
#    input:
#        com_file=LFR_COM_FILE,
#        ref_com_file=SIM_LFR_COM_FILE,
#    output:
#        output_file=LFR_COM_DETECT_FILE,
#    params:
#        K=1,
#    script:
#        "workflow/eval-detected-community.py"
#
#
#rule concat_auc_result_lfr_file:
#    input:
#        input_files=LFR_AUC_FILE_ALL,
#    output:
#        output_file=LFR_AUC_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_kmeans_result_lfr_file:
#    input:
#        input_files=LFR_KMEANS_FILE_ALL,
#    output:
#        output_file=LFR_KMEANS_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_community_detection_result_lfr_file:
#    input:
#        input_files=LFR_COM_DETECT_FILE_ALL,
#    output:
#        output_file=LFR_COM_DETECT_RES_FILE,
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_dist_result_lfr_file:
#    input:
#        input_files=LFR_DIST_FILE_ALL,
#    output:
#        output_file=LFR_DIST_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule concat_separatability_result_lfr_file:
#    input:
#        input_files=LFR_SEPARATABILITY_FILE_ALL,
#    output:
#        output_file=LFR_SEPARATABILITY_RES_FILE,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/concat-files.py"
#
#
#rule eval_lfr_distances:
#    input:
#        emb_file=LFR_EMB_FILE,
#        com_file=SIM_LFR_COM_FILE,
#    output:
#        output_file=LFR_DIST_FILE,
#    params:
#        K=1,
#        num_samples=10000,
#    wildcard_constraints:
#        model_name="("
#        + ")|(".join(emb_params["model_name"] + emb_params_rw["model_name"])
#        + ")",
#    script:
#        "workflow/eval-community-distances.py"
#
#
#rule eval_lfr_separatability:
#    input:
#        emb_file=LFR_EMB_FILE,
#        com_file=SIM_LFR_COM_FILE,
#    output:
#        output_file=LFR_SEPARATABILITY_FILE,
#    params:
#        K=1
#    script:
#        "workflow/eval-community-separatability.py"
#
#rule plot_distances:
#    input:
#        input_file = DIST_RES_FILE
#    params:
#        cdiff = lambda wildcards: float(wildcards.cdiff),
#        cave = lambda wildcards: float(wildcards.cave),
#        metric = lambda wildcards: wildcards.metric,
#        dim = lambda wildcards: float(wildcards.dim),
#        K = lambda wildcards: "None" if wildcards.K == "None" else int(wildcards.K)
#    output:
#        output_file = FIG_DIST_RES
#    script:
#        "workflow/plotter/plot-distance.py"
#
#rule plot_separatability:
#    input:
#        input_file = SEPARATABILITY_RES_FILE
#    params:
#        cdiff = lambda wildcards: float(wildcards.cdiff),
#        cave = lambda wildcards: float(wildcards.cave),
#        metric = lambda wildcards: wildcards.metric,
#        dim = lambda wildcards: float(wildcards.dim),
#        K = lambda wildcards: "None" if wildcards.K == "None" else int(wildcards.K)
#    output:
#        output_file = FIG_SEP_RES
#    script:
#        "workflow/plotter/plot-separatability.py"
#
#rule plot_distance_separatability:
#    input:
#        two_coms_euc = FIG_DIST_RES.format(data="multi_coms", metric = "euclidean", cave=50, cdiff="100", dim=64, K = 2),
#        two_coms_cos = FIG_DIST_RES.format(data="multi_coms", metric = "cosine", cave=50, cdiff="100", dim=64, K = 2),
#        two_coms_sep = FIG_SEP_RES.format(data="multi_coms", metric = "None", cave=50, cdiff="100", dim=64, K = 2),
#        multi_coms_euc = FIG_DIST_RES.format(data="multi_coms", metric = "euclidean", cave="100", cdiff=160, dim=64, K = 50),
#        multi_coms_cos = FIG_DIST_RES.format(data="multi_coms", metric = "cosine", cave=50, cdiff="100", dim=64, K = 50),
#        multi_coms_sep = FIG_SEP_RES.format(data="multi_coms", metric = "None", cave=50, cdiff="100", dim=64, K = 50),
#        multi_fs_coms_euc = FIG_DIST_RES.format(data="multi_fixed_size_coms", metric = "euclidean", cave=50, cdiff="100", dim=64, K = None),
#        multi_fs_coms_cos = FIG_DIST_RES.format(data="multi_fixed_size_coms", metric = "cosine", cave=50, cdiff="100", dim=64, K = None),
#        multi_fs_coms_sep = FIG_SEP_RES.format(data="multi_fixed_size_coms", metric = "None", cave=50, cdiff="100", dim=64, K = None),
#        roc_coms_euc = FIG_DIST_RES.format(data="ring_of_cliques", metric = "euclidean", cave=50, cdiff="100", dim=64, K = None),
#        roc_coms_cos = FIG_DIST_RES.format(data="ring_of_cliques", metric = "cosine", cave=50, cdiff="100", dim=64, K = None),
#        roc_coms_sep = FIG_SEP_RES.format(data="ring_of_cliques", metric = "None", cave=50, cdiff="100", dim=64, K = None),
#    #output:
#    #    FIG_DIST_SEP_RES,
#    #shell:
#    #    "pdfjam --nup 3x3 --frame false --delta '1mm 1mm' --no-landscape --outfile {output} {input} "
#
#
#
##rule plot_distance
#
#
##
## Misc
##
#rule all:
#    input:
#        ##MULTI_COM_SEPARATABILITY_FILE_ALL,
#        # Infomap
#        ##MULTI_COM_COM_DETECT_RES_FILE,
#        #MULTI_FIXED_SZ_COM_COM_DETECT_RES_FILE,
#        LFR_COM_DETECT_RES_FILE,
#        #RING_OF_CLIQUE_COM_DETECT_RES_FILE,
#        # Separability
#        ##MULTI_COM_SEPARATABILITY_RES_FILE,
#        #MULTI_FIXED_SZ_COM_SEPARATABILITY_RES_FILE,
#        #RING_OF_CLIQUE_SEPARATABILITY_RES_FILE,
#        LFR_SEPARATABILITY_RES_FILE,
#        # Distance ratio
#        #MULTI_COM_DIST_RES_FILE,
#        #MULTI_FIXED_SZ_COM_DIST_RES_FILE,
#        #RING_OF_CLIQUE_DIST_RES_FILE,
#        #LFR_DIST_RES_FILE,
#        # Kmeans
#        ##MULTI_FIXED_SZ_COM_KMEANS_RES_FILE,
#        #MULTI_COM_KMEANS_RES_FILE,
#        #RING_OF_CLIQUE_KMEANS_RES_FILE,
#        LFR_KMEANS_RES_FILE,
#        #expand(SEPARATABILITY_RES_FILE, data = DATA_LIST),
#        #MULTI_COM_COM_DETECT_RES_FILE,
#        #MULTI_FIXED_SZ_COM_COM_DETECT_RES_FILE,
#
#rule _all:
#    input:
#        #MULTI_COM_COM_DETECT_RES_FILE,
#        #MULTI_FIXED_SZ_COM_COM_DETECT_RES_FILE,
#        #RING_OF_CLIQUE_COM_DETECT_RES_FILE,
#
#rule __all:
#    input:
#        #MULTI_FIXED_SZ_COM_FILE_ALL,
##        LFR_SEPARATABILITY_RES_FILE,
##        LFR_DIST_RES_FILE,
##        RING_OF_CLIQUE_DIST_RES_FILE,
##        RING_OF_CLIQUE_SEPARATABILITY_RES_FILE,
##        MULTI_FIXED_SZ_COM_DIST_RES_FILE,
##        MULTI_FIXED_SZ_COM_SEPARATABILITY_RES_FILE,
##        MULTI_COM_DIST_RES_FILE,
##        MULTI_COM_SEPARATABILITY_RES_FILE,
#        #MULTI_COM_COM_DETECT_RES_FILE,
#        #MULTI_FIXED_SZ_COM_COM_DETECT_RES_FILE,
#        #MULTI_FIXED_SZ_COM_KMEANS_RES_FILE,
#        #MULTI_COM_KMEANS_RES_FILE,
#        #RING_OF_CLIQUE_COM_DETECT_RES_FILE,
#        #RING_OF_CLIQUE_KMEANS_RES_FILE
#        #MULTI_COM_KMEANS_FILE_ALL
#        #TWO_COM_EMB_FILE_ALL, #SIM_TWO_COM_NET_ALL
#        #TWO_COM_SIM_FILE,RES_TWO_COM_KMEANS_FILE
#        #TWO_COM_EMB_FILE_ALL
#
