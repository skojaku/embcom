# =========
# FIGURES
# =========
fig_params_perf_vs_mixing = {
    "q": [2, 64],
    "dim": [64],
    "n": [1000, 10000, 100000],
    "metric": ["cosine"],
    "length": [10],
    "clustering": ["voronoi"],
    "score_type": ["esim"],
    "cave": [10, 50],
    "data": ["multi_partition_model"],
}
fig_perf_vs_mixing_paramspace = to_paramspace(fig_params_perf_vs_mixing)
FIG_PERFORMANCE_VS_MIXING = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_{fig_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)

# ================================
# Networks and communities
# ================================

net_params = {
    "n": [100000],  # Network size
    "K": [2, 64],  # Number of communities
    "cave": [10, 50],  # average degree
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
eval_params = {
    "scoreType": ["esim", "nmi"],
}
eva_emb_paramspace = to_paramspace(
    [eval_params, net_params, emb_params, clustering_params]
)
EVAL_EMB_FILE = j(EVA_DIR, f"score_clus_{eva_emb_paramspace.wildcard_pattern}.npz")

eva_paramspace = to_paramspace([eval_params, net_params, com_detect_params])
EVAL_FILE = j(EVA_DIR, f"score_{eva_paramspace.wildcard_pattern}.npz")


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
    params:
        parameters=eva_paramspace.instance,
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
    params:
        parameters=eva_paramspace.instance,
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
            **eval_params
        ) + expand(
            EVAL_EMB_FILE,
            data="multi_partition_model",
            **net_params,
            **emb_params,
            **clustering_params,
            **eval_params
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
        input_file=EVAL_CONCAT_FILE,
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING,
    params:
        parameters=fig_perf_vs_mixing_paramspace.instance,
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
