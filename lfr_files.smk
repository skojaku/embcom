# ================================
# Networks and communities
# ================================

lfr_net_params = {
    #"n": [1000],  # Network size
    "n": [10000],  # Network size
    #"n": [1000, 10000, 100000],  # Network size
    "k": [5, 10, 50],  # Average degree
    "tau": [3],  # degree exponent
    "tau2": [1],  # community size exponent
    "minc": [50],  # min community size
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(3),  # Number of samples
}

# Convert to a paramspace
lfr_net_paramspace = to_paramspace(lfr_net_params)
LFR_NET_FILE = j(NET_DIR, f"net_{lfr_net_paramspace.wildcard_pattern}.npz")
LFR_NODE_FILE = j(NET_DIR, f"node_{lfr_net_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================
lfr_emb_paramspace = to_paramspace([lfr_net_params, emb_params])
LFR_EMB_FILE = j(EMB_DIR, f"{lfr_emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
lfr_com_detect_paramspace = to_paramspace([lfr_net_params, com_detect_params])

# Community detection
LFR_COM_DETECT_FILE = j(COM_DIR, f"{lfr_com_detect_paramspace.wildcard_pattern}.npz")

# Community detection by clustering to embedding
lfr_com_detect_emb_paramspace = to_paramspace([lfr_net_params, emb_params, clustering_params])
LFR_COM_DETECT_EMB_FILE = j(
    COM_DIR, f"clus_{lfr_com_detect_emb_paramspace.wildcard_pattern}.npz"
)


# ==========
# Evaluation
# ==========
LFR_EVAL_EMB_FILE = j(EVA_DIR, f"score_clus_{lfr_com_detect_emb_paramspace.wildcard_pattern}.npz")
LFR_EVAL_FILE = j(EVA_DIR, f"score_{lfr_com_detect_paramspace.wildcard_pattern}.npz")


# =========
# FIGURES
# =========
fig_lfr_params_perf_vs_mixing = {
    "dim": [64],
    "k": [5, 10, 50, 100],  # Average degree
    #"n": [1000],
    "n": [100000],
    #"dim":[64, 256],
    "metric": ["cosine"],
    "length": [10],
    "clustering": ["voronoi"],
    #"clustering": ["voronoi", "kmeans", "birch"],
    "score_type": ["esim"],
    "data": ["lfr"],
}
fig_lfr_perf_vs_mixing_paramspace = to_paramspace(fig_lfr_params_perf_vs_mixing)
FIG_LFR_PERFORMANCE_VS_MIXING = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_{fig_lfr_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)

# ======
# RULES
# ======
#
# network generation
#
rule generate_lfr_net:
    params:
        parameters=lfr_net_paramspace.instance,
    output:
        output_file=LFR_NET_FILE,
        output_node_file=LFR_NODE_FILE,
    wildcard_constraints:
        data="lfr"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/net_generator/generate-lfr-net.py"

#
# Embedding
#
use rule embedding_multi_partition_model as embedding_lfr with:
    input:
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EMB_FILE,
    params:
        parameters=lfr_emb_paramspace.instance,


#
# Clustering
#
use rule voronoi_clustering_multi_partition_model as voronoi_clustering_lfr with:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=lfr_com_detect_emb_paramspace.instance,

use rule birch_best_clustering_multi_partition_model as birch_best_clustering_lfr with:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=lfr_com_detect_emb_paramspace.instance,
        n_clusters = "true"

use rule birch_clustering_multi_partition_model as birch_clustering_lfr with:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=lfr_com_detect_emb_paramspace.instance,
        n_clusters = "data"

use rule kmeans_clustering_multi_partition_model as kmeans_clustering_lfr with:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=lfr_com_detect_emb_paramspace.instance,


use rule community_detection_multi_partition_model as community_detection_lfr with:
    input:
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_FILE,
    params:
        parameters=lfr_com_detect_paramspace.instance,


#
# Evaluation
#
use rule evaluate_communities as evaluate_communities_lfr with:
    input:
        detected_group_file=LFR_COM_DETECT_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_FILE,

use rule evaluate_communities_for_embedding as evaluate_communities_for_embedding_lfr with:
    input:
        detected_group_file=LFR_COM_DETECT_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_EMB_FILE,

rule concatenate_results_lfr:
    input:
        input_files = expand(LFR_EVAL_FILE, data="lfr", **lfr_net_params, **com_detect_params) + expand(LFR_EVAL_EMB_FILE, data="lfr", **lfr_net_params, **emb_params, **clustering_params)
    output:
        output_file=EVAL_CONCAT_FILE
    wildcard_constraints:
        data="lfr"
    params:
        to_int=["n", "k", "tau", "tau2", "minc", "dim", "sample", "length", "dim"],
        to_float=["mu"],
    script:
        "workflow/evaluation/concatenate_results.py"

#
# Plot
#
rule plot_lfr_performance_vs_mixing:
    input:
        input_file="data/lfr/all-result.csv",
        #input_file=EVAL_CONCAT_FILE,
    output:
        output_file=FIG_LFR_PERFORMANCE_VS_MIXING,
    params:
        parameters=fig_lfr_perf_vs_mixing_paramspace.instance,
        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items()]),
        model_names = ["node2vec", "deepwalk", "line", "linearized-node2vec", "modspec", "leigenmap", "non-backtracking", "bp", "infomap", "flatsbm" ],
        with_legend = lambda wildcards: "True" if str(wildcards.k)=="5" else "False"
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/plot/plot-mixing-vs-performance-lfr.py"

rule plot_lfr_performance_vs_mixing_all:
    input:
        expand(FIG_LFR_PERFORMANCE_VS_MIXING, **fig_lfr_params_perf_vs_mixing),
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING_ALL.format(data="lfr"),
    run:
        shell("pdfjam {input} --nup 3x4 --suffix 3up --outfile {output}")
