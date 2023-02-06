# =========
# FIGURES
# =========
fig_params_perf_vs_mixing = {
    "q": [2, 50],
    "dim": [64],
    "n": [10000],
    "metric": ["cosine"],
    "length": [10],
    "clustering": ["voronoi"],
    "score_type": ["esim"],
    "cave": [5],
    "data": ["multi_partition_model"],
}
fig_perf_vs_mixing_paramspace = to_paramspace(fig_params_perf_vs_mixing)
FIG_PERFORMANCE_VS_MIXING = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_{fig_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)
FIG_PERFORMANCE_VS_MIXING_NB = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_nonbacktracking_{fig_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)

# ================================
# Networks and communities
# ================================

net_params = {
    #"n": [10000],  # Network size
    "n": [10000, 100000],  # Network size
    #"n": [100000, 10000],
    #"n": [1000, 10000, 100000],
    "K": [2, 50],  # Number of communities
    #"K": [2, 64],  # Number of communities
    "cave": [5, 10, 50],  # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(3),  # Number of samples
}

# Convert to a paramspace
net_paramspace = to_paramspace(net_params)
NET_FILE = j(NET_DIR, f"net_{net_paramspace.wildcard_pattern}.npz")
NODE_FILE = j(NET_DIR, f"node_{net_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================

emb_paramspace = to_paramspace([net_params, emb_params])
EMB_FILE = j(EMB_DIR, f"{emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
com_detect_paramspace = to_paramspace([net_params, com_detect_params])

# Community detection
COM_DETECT_FILE = j(COM_DIR, f"{com_detect_paramspace.wildcard_pattern}.npz")

# Community detection by clustering to embedding
com_detect_emb_paramspace = to_paramspace([net_params, emb_params, clustering_params])
COM_DETECT_EMB_FILE = j(
    COM_DIR, f"clus_{com_detect_emb_paramspace.wildcard_pattern}.npz"
)


# ==========
# Evaluation
# ==========
EVAL_EMB_FILE = j(EVA_DIR, f"score_clus_{com_detect_emb_paramspace.wildcard_pattern}.npz")
EVAL_FILE = j(EVA_DIR, f"score_{com_detect_paramspace.wildcard_pattern}.npz")


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
FIG_SPECTRAL_DENSITY_FILE = j(
    FIG_DIR, "spectral-density", f"{bipartition_paramspace.wildcard_pattern}.pdf"
)


# ======
# RULES
# ======
#
# network generation
#
rule generate_net_multi_partition_model:
    params:
        parameters=net_paramspace.instance,
    output:
        output_file=NET_FILE,
        output_node_file=NODE_FILE,
    wildcard_constraints:
        data="multi_partition_model",
    resources:
        mem="12G",
        time="04:00:00"
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
    resources:
        mem="12G",
        time="01:00:00"
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
    resources:
        mem="12G",
        time="01:00:00"
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
        "workflow/community-detection/detect-community.py"


#
# Evaluation
#
rule evaluate_communities:
    input:
        detected_group_file=COM_DETECT_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_FILE,
    resources:
        mem="12G",
        time="00:10:00"
    script:
        "workflow/evaluation/eval-com-detect-score.py"


rule evaluate_communities_for_embedding:
    input:
        detected_group_file=COM_DETECT_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_EMB_FILE,
    resources:
        mem="12G",
        time="00:20:00"
    script:
        "workflow/evaluation/eval-com-detect-score.py"


rule concatenate_results_multipartition:
    input:
        input_files=expand(
            EVAL_FILE,
            data="multi_partition_model",
            **net_params,
            **com_detect_params,
        ) + expand(
            EVAL_EMB_FILE,
            data="multi_partition_model",
            **net_params,
            **emb_params,
            **clustering_params,
        ),
    output:
        output_file=EVAL_CONCAT_FILE,
    params:
        to_int=["n", "K", "dim", "sample", "length", "dim", "cave"],
        to_float=["mu"],
    wildcard_constraints:
        data="multi_partition_model",
    resources:
        mem="4G",
        time="00:50:00"
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
rule plot_performance_vs_mixing:
    input:
        #input_file=EVAL_CONCAT_FILE,
        input_file="data/multi_partition_model/all-result.csv",
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING,
    params:
        parameters=fig_perf_vs_mixing_paramspace.instance,
        model_names = ["non-backtracking-node2vec", "nonbacktracking", "node2vec", "deepwalk", "depthfirst-node2vec", "non-backtracking-deepwalk", "line", "infomap", "flatsbm", "torch-node2vec", "torch-modularity", "modspec", "linearized-node2vec"]
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/plot/plot-mixing-vs-performance.py"

rule plot_performance_vs_mixing_nb:
    input:
        #input_file="data/multi_partition_model/all-result.csv",
        input_file=EVAL_CONCAT_FILE,
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING_NB,
    params:
        parameters=fig_perf_vs_mixing_paramspace.instance,
        model_names = ["non-backtracking-node2vec", "nonbacktracking", "depthfirst-node2vec", "non-backtracking-deepwalk"]
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/plot/plot-mixing-vs-performance.py"


rule plot_spectral_density:
    input:
        input_file=SPECTRAL_DENSITY_FILE,
    output:
        output_file=FIG_SPECTRAL_DENSITY_FILE,
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/plot/plot-spectral-density.py"
