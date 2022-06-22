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
    "n": [2500, 5000, 10000, 50000, 100000, 500000],  # Network size
    "K": [2, 16, 32, 64, 128],  # Number of communities
    "cave": [10, 20, 50],  # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(3),  # Number of samples
}

net_params = {
    "n": [2500, 5000, 10000, 50000, 100000], # Network size
    "K": [2, 16, 32, 64, 128],  # Number of communities
    "cave": [10, 20, 50],  # average degree
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
emb_params = {
    "model_name": [
        "node2vec",
        "deepwalk",
        "leigenmap",
        "modspec",
        #"levy-word2vec",
        "linearized-node2vec",
        "non-backtracking-node2vec",
    ],
    # "model_name": ["node2vec", "glove", "depthfirst-node2vec"],
    # "model_name": ["leigenmap", "modspec", "nonbacktracking"],
    "window_length": [1,10],
    "dim": [0, 64],
}


emb_paramspace = to_paramspace([net_params, emb_params])

EMB_FILE = j(EMB_DIR, f"{emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
com_detect_params = {
    "model_name": ["infomap"],
    #"model_name": ["infomap", "flatsbm"],
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

EVAL_CONCAT_FILE = j(EVA_DIR, f"all-result.csv")


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
