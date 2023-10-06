# =========
# FIGURES
# =========
fig_params_perf_vs_mixing_robustness = {
    "q": [2, 50],
    "dim": [64],
    "n": [10000],
    "metric": ["cosine"],
    "length": [10],
    "clustering": ["voronoi"],
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
    "nWalks":[1, 5, 10, 20, 40, 80, 160],
    "dim": [64],
}

robustness_emb_paramspace = to_paramspace([net_params, robustness_emb_params])
ROBUSTNESS_EMB_FILE = j(EMB_DIR, "robustness", f"{robustness_emb_paramspace.wildcard_pattern}.npz")

com_detect_robustness_emb_paramspace = to_paramspace([net_params, robustness_emb_params, clustering_params])
EVAL_ROBUSTNESS_EMB_FILE = j(EVA_DIR, f"score_clus_{com_detect_robustness_emb_paramspace.wildcard_pattern}.npz")
COM_DETECT_ROBUSTNESS_EMB_FILE = j(
    COM_DIR, f"clus_{com_detect_robustness_emb_paramspace.wildcard_pattern}.npz"
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


rule concatenate_results_robustness:
    input:
        expand(
            EVAL_ROBUSTNESS_EMB_FILE,
            data="multi_partition_model",
            **net_params,
            **robustness_emb_params,
            **clustering_params,
        ),
    output:
        output_file=EVAL_CONCAT_FILE,
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