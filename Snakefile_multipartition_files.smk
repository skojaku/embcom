# =========
# FIGURES
# =========
fig_params_perf_vs_mixing = {
    "q": [2, 50],
    "dim": [16, 32, 64, 128],
    #"n": [1000],
    "n": [10000, 100000],
    "metric": ["cosine"],
    "length": [10],
    #"clustering": ["voronoi"],
    "clustering": ["voronoi", "kmeans"],
    #""clustering": ["voronoi", "kmeans", "birch"],
    "score_type": ["esim"],
    "cave": [5, 10, 50],
    "data": ["multi_partition_model"],
}
fig_perf_vs_mixing_paramspace = to_paramspace(fig_params_perf_vs_mixing)
FIG_PERFORMANCE_VS_MIXING = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_{fig_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)
FIG_PERFORMANCE_VS_MIXING_SPEC_VS_SGD = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_spec_vs_sgd_{fig_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)

# ================================
# Networks and communities
# ================================

net_params = {
    #"n": [1000],  # Network size
    "n": [100000],  # Network size
    #"n": [10000, 100000],  # Network size
    #"n": [1000, 10000],
    #"n": [1000, 10000, 100000],
    "K": [2],  # Number of communities
    #"K": [2, 50],  # Number of communities
    "cave": [5],  # average degree
    #"cave": [5, 10, 50],  # average degree
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
params_loss_landscape = {
    "n":10000,
    "K":50,
    "cave":10,
    "mu":0.2,
    "sample":0,
    "data":"multi_partition_model"
}
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

#
# Loss landscape
#
LOSS_LANDSCAPE_MODEL_LIST = ["modularity", "laplacian"]
FIG_LOSS_LANDSCAPE = j("figs", "loss_landscape", "loss_landscape_model~{model}.pdf")

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

rule hdbscan_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="hdbscan",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/hdbscan-clustering.py"

rule birch_best_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
        n_clusters = "true"
    wildcard_constraints:
        clustering="birch-best",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/birch-clustering.py"


rule birch_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
        n_clusters = "data"
    wildcard_constraints:
        clustering="birch",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/birch-clustering.py"

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
        dimThreshold= False,
        normalize= False,
        model_names = ["node2vec", "deepwalk", "line", "modspec", "leigenmap", "nonbacktracking", "bp", "infomap", "flatsbm" ],
        #title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items()]),
        with_legend = lambda wildcards: "True" if str(wildcards.cave)=="5" else "False"
    resources:
        mem="4G",
        time="00:50:00"
    script:
        "workflow/plot/plot-mixing-vs-performance.py"

rule plot_performance_vs_mixing_mod_vs_spec:
    input:
        input_file="data/multi_partition_model/all-result.csv",
        #input_file=EVAL_CONCAT_FILE,
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING_SPEC_VS_SGD,
    params:
        parameters=fig_perf_vs_mixing_paramspace.instance,
        dimThreshold= False,
        normalize= False,
        model_names = ["node2vec", "modspec", "leigenmap", "torch-modularity", "torch-laplacian-eigenmap", "bp"],
        #title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items()]),
        with_legend = lambda wildcards: "True" if str(wildcards.cave)=="5" else "False"
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

rule plot_performance_vs_mixing_all:
    input:
        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing),
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING_ALL.format(data="multi_partition_model"),
    run:
        shell("pdfjam {input} --nup 3x4 --suffix 3up --outfile {output}")

rule plot_loss_landscape_modularity:
    input:
        net_file = NET_FILE.format(**params_loss_landscape),
        node_table_file = NODE_FILE.format(**params_loss_landscape)
    output:
        output_file = FIG_LOSS_LANDSCAPE,
    params:
        **params_loss_landscape
    wildcard_constraints:
        model = "modularity"
    script:
        "workflow/plot/plot_modularity_loss_landscape.py"

rule plot_loss_landscape_laplacian:
    input:
        net_file = NET_FILE.format(**params_loss_landscape),
        node_table_file = NODE_FILE.format(**params_loss_landscape)
    output:
        output_file = FIG_LOSS_LANDSCAPE,
    params:
        **params_loss_landscape
    wildcard_constraints:
        model = "laplacian"
    script:
        "workflow/plot/plot_laplacian_loss_landscape.py"
