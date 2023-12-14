# =========
# FIGURES
# =========
fig_params_perf_vs_mixing_robustness = {
    "q": [2, 50],
    "dim": [64],
    "n": [10000],
    #"n": [100000],
    "metric": ["cosine"],
    "length": [10],
    "clustering": ["voronoi", "kmeans"],
    "score_type": ["esim"],
    "cave": [5, 10, 50],
    "data": ["multi_partition_model", "lfr"],
}
fig_perf_vs_mixing_robustness_paramspace = to_paramspace(fig_params_perf_vs_mixing_robustness)
FIG_PERFORMANCE_VS_MIXING_VS_TRAIN_ITER = j(
    FIG_DIR,
    "robustness_check",
    f"fig_{fig_perf_vs_mixing_robustness_paramspace.wildcard_pattern}.pdf",
)
# =================
# Embedding
# =================
robustness_emb_params = {
    "model_name": [
        "node2vec",
        "deepwalk",
        #"glove",
        #"line",
        #"leigenmap",
        #"modspec",
        #"linearized-node2vec",
    ],
    "window_length": [10],
    #"window_length": [1, 3, 10, 15],
    "nWalks":[40],
    #"nWalks":[1, 5, 40, 80],
    #"nWalks":[1, 5, 10, 20, 40, 80, 160],
    "dim": [512],
    #"dim": [8, 64, 128, 256, 512],
}
# -------
# @ network
# --------
robustness_net_params = {
    "n": [10000],  # Network size
    #"n": [100000],  # Network size
    "K": [2, 50],  # Number of communities
    "cave": [5, 10, 50],  # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(3),  # Number of samples
}

robustness_lfr_net_params = {
    "n": [10000],  # Network size
    "k": [5, 10, 50],  # Average degree
    "tau": [3],  # degree exponent
    "tau2": [1],  # community size exponent
    "minc": [50],  # min community size
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(3),  # Number of samples
}
# -------

robustness_emb_paramspace = to_paramspace([robustness_net_params, robustness_emb_params])
ROBUSTNESS_EMB_FILE = j(EMB_DIR, "robustness", f"{robustness_emb_paramspace.wildcard_pattern}.npz")

com_detect_robustness_emb_paramspace = to_paramspace([robustness_net_params, robustness_emb_params, clustering_params])
EVAL_ROBUSTNESS_EMB_FILE = j(EVA_DIR, "robustness", f"score_clus_{com_detect_robustness_emb_paramspace.wildcard_pattern}.npz")
COM_DETECT_ROBUSTNESS_EMB_FILE = j(
    COM_DIR, "robustness", f"clus_{com_detect_robustness_emb_paramspace.wildcard_pattern}.npz"
)


robustness_lfr_emb_paramspace = to_paramspace([robustness_lfr_net_params, robustness_emb_params])
ROBUSTNESS_LFR_EMB_FILE = j(EMB_DIR, "robustness", f"{robustness_lfr_emb_paramspace.wildcard_pattern}.npz")

lfr_com_detect_robustness_emb_paramspace = to_paramspace([robustness_lfr_net_params, robustness_emb_params, clustering_params])
EVAL_ROBUSTNESS_LFR_EMB_FILE = j(EVA_DIR, "robustness", f"score_clus_{lfr_com_detect_robustness_emb_paramspace.wildcard_pattern}.npz")
COM_DETECT_ROBUSTNESS_LFR_EMB_FILE = j(
    COM_DIR, "robustness", f"clus_{lfr_com_detect_robustness_emb_paramspace.wildcard_pattern}.npz"
)


# ======
# RULES
# ======
#
# Embedding
#
rule embedding_multi_partition_model_robustness:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=ROBUSTNESS_EMB_FILE,
    params:
        parameters=robustness_emb_paramspace.instance,
    script:
        "workflow/embedding/embedding.py"

rule embedding_multi_partition_model_robustness_lfr:
    input:
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=ROBUSTNESS_LFR_EMB_FILE,
    params:
        parameters=robustness_lfr_emb_paramspace.instance,
    script:
        "workflow/embedding/embedding.py"


#
# Clustering
#
rule voronoi_clustering_multi_partition_model_robustness:
    input:
        emb_file=ROBUSTNESS_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_ROBUSTNESS_EMB_FILE,
    params:
        parameters=com_detect_robustness_emb_paramspace.instance,
    wildcard_constraints:
        clustering="voronoi",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/voronoi-clustering.py"


rule kmeans_clustering_multi_partition_model_robustness:
    input:
        emb_file=ROBUSTNESS_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_ROBUSTNESS_EMB_FILE,
    params:
        parameters=com_detect_robustness_emb_paramspace.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/kmeans-clustering.py"

rule hdbscan_clustering_multi_partition_model_robustness:
    input:
        emb_file=ROBUSTNESS_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_ROBUSTNESS_EMB_FILE,
    params:
        parameters=com_detect_robustness_emb_paramspace.instance,
    wildcard_constraints:
        clustering="hdbscan",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/hdbscan-clustering.py"

rule voronoi_clustering_lfr_robustness_lfr:
    input:
        emb_file=ROBUSTNESS_LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=COM_DETECT_ROBUSTNESS_LFR_EMB_FILE,
    params:
        parameters=lfr_com_detect_robustness_emb_paramspace.instance,
    wildcard_constraints:
        clustering="voronoi",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/voronoi-clustering.py"


rule kmeans_clustering_multi_partition_model_robustness_lfr:
    input:
        emb_file=ROBUSTNESS_LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=COM_DETECT_ROBUSTNESS_LFR_EMB_FILE,
    params:
        parameters=lfr_com_detect_robustness_emb_paramspace.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/kmeans-clustering.py"

rule hdbscan_clustering_multi_partition_model_robustness_lfr:
    input:
        emb_file=ROBUSTNESS_LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=COM_DETECT_ROBUSTNESS_LFR_EMB_FILE,
    params:
        parameters=lfr_com_detect_robustness_emb_paramspace.instance,
    wildcard_constraints:
        clustering="hdbscan",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/hdbscan-clustering.py"
#
# Evaluation
#
rule evaluate_communities_for_embedding_robustness:
    input:
        detected_group_file=COM_DETECT_ROBUSTNESS_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_ROBUSTNESS_EMB_FILE,
    resources:
        mem="12G",
        time="00:20:00"
    script:
        "workflow/evaluation/eval-com-detect-score.py"

rule evaluate_communities_lfr_for_embedding_robustness:
    input:
        detected_group_file=COM_DETECT_ROBUSTNESS_LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=EVAL_ROBUSTNESS_LFR_EMB_FILE,
    resources:
        mem="12G",
        time="00:20:00"
    script:
        "workflow/evaluation/eval-com-detect-score.py"


rule concatenate_results_robustness:
    input:
        expand(
            EVAL_ROBUSTNESS_EMB_FILE,
            data="multi_partition_model",
            **robustness_net_params,
            **robustness_emb_params,
            **clustering_params,
        ),
    output:
        output_file=EVAL_CONCAT_ROBUSTNESS_FILE,
    params:
        to_int=["n", "K", "dim", "sample", "length", "dim", "cave", "num_walks"],
        to_float=["mu"],
    wildcard_constraints:
        data="multi_partition_model",
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/evaluation/concatenate_results.py"

rule concatenate_results_robustness_lfr:
    input:
        expand(
            EVAL_ROBUSTNESS_LFR_EMB_FILE,
            data="lfr",
            **robustness_lfr_net_params,
            **robustness_emb_params,
            **clustering_params,
        ),
    output:
        output_file=EVAL_CONCAT_ROBUSTNESS_FILE,
    params:
        to_int=["n", "k", "tau", "tau2", "minc", "dim", "sample", "length", "dim"],
        to_float=["mu", "tau"],
    wildcard_constraints:
        data="lfr",
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/evaluation/concatenate_results.py"

rule robustness_all:
    input:
        expand(EVAL_CONCAT_ROBUSTNESS_FILE, data = "multi_partition_model"),
        expand(EVAL_CONCAT_ROBUSTNESS_FILE, data = "lfr")
##
## Plot
##
#rule plot_performance_vs_mixing:
#    input:
#        #input_file=EVAL_CONCAT_FILE,
#        input_file="data/multi_partition_model/all-result.csv",
#    output:
#        output_file=FIG_PERFORMANCE_VS_MIXING,
#    params:
#        parameters=fig_perf_vs_mixing_paramspace.instance,
#        dimThreshold= False,
#        normalize= False,
#        model_names = ["node2vec", "deepwalk", "line", "modspec", "leigenmap", "nonbacktracking", "bp", "infomap", "flatsbm" ],
#        #model_names = ["node2vec", "deepwalk", "line", "linearized-node2vec", "modspec", "leigenmap", "nonbacktracking", "bp", "infomap", "flatsbm" ],
#        title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items()]),
#        with_legend = lambda wildcards: "True" if str(wildcards.cave)=="5" else "False"
#    resources:
#        mem="4G",
#        time="00:50:00"
#    script:
#        "workflow/plot/plot-mixing-vs-performance.py"
