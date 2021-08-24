import numpy as np
from os.path import join as j


configfile: "workflow/config.yaml"


DATA_DIR = config["data_dir"]
FIG_DIR = "figs"
RES_DIR = j(DATA_DIR, "results")

PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

DERIVED_DIR = j(DATA_DIR, "derived")
SIM_R_DIR = j(DERIVED_DIR, "sim_R")
SIM_R_RES = j(SIM_R_DIR, "rvals.csv")

TWO_COM_NET_DIR = j(DATA_DIR, "networks", "two_coms")
TWO_COM_EMB_DIR = j(DATA_DIR, "embeddings", "two_coms")
sim_net_params = {
    "n": [50, 100, 500, 1000, 10000, 100000],
    #"n": [50, 100, 500, 1000, 10000, 100000, 1000000],
    "cin": [10, 12, 14, 16, 18, 20, 30, 40],
    "cout": [5],
    "sample": np.arange(10),
}
SIM_TWO_COM_NET = j(
    TWO_COM_NET_DIR, "net_n={n}_cin={cin}_cout={cout}_sample={sample}.npz"
)
SIM_TWO_COM_NET_ALL = expand(SIM_TWO_COM_NET, **sim_net_params)

TWO_COM_EMB_FILE_DIR = j(TWO_COM_EMB_DIR, "embeddings")
emb_params_rw = {  # parameter for methods baesd on random walks
    "model_name": ["node2vec", "glove"],
    "window_length": [3, 5, 10],
    "dim": [1, 2, 8, 32, 128],
}
emb_params = {
    "model_name": ["leigenmap", "levy-word2vec", "adjspec", "modspec"],
    "window_length": [10],
    "dim": [1, 2, 8, 32, 128],
}
TWO_COM_EMB_FILE = j(
    TWO_COM_EMB_FILE_DIR,
    "embnet_n={n}_cin={cin}_cout={cout}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
)
TWO_COM_EMB_FILE_ALL = expand(
    TWO_COM_EMB_FILE, **sim_net_params, **emb_params
) + expand(TWO_COM_EMB_FILE, **sim_net_params, **emb_params_rw)
TWO_COM_AUC_FILE = j(RES_DIR, "two_coms", "auc.csv")
TWO_COM_SIM_FILE = j(RES_DIR, "two_coms", "sim_vals.csv")
RES_TWO_COM_KMEANS_FILE = j(RES_DIR, "two_coms", "res-kmeans.csv")
FIG_TWO_COM_AUC = j(FIG_DIR, "two-coms-auc.pdf")
FIG_SIM_WIJ = j(FIG_DIR, "rvals.pdf")


rule all:
    input:
        PAPER,
        SUPP,


rule paper:
    input:
        PAPER_SRC,
        SUPP_SRC,
    params:
        paper_dir=PAPER_DIR,
    output:
        PAPER,
        SUPP,
    shell:
        "cd {params.paper_dir}; make"


rule sample_Wij_entry:
    params:
        cin=30,
        cout=5,
        num_sample=100,
    output:
        output_file=SIM_R_RES,
    script:
        "workflow/simulate-R-matrix.py"


rule plot_Wij_entry:
    params:
        input_file=SIM_R_RES,
    output:
        output_file=FIG_SIM_WIJ,
    script:
        "workflow/plot-rvals.py"


rule generate_2com_net:
    params:
        cin=lambda wildcards: int(wildcards.cin),
        cout=lambda wildcards: int(wildcards.cout),
        n=lambda wildcards: int(wildcards.n),
    output:
        output_file=SIM_TWO_COM_NET,
    script:
        "workflow/generate-2com-net.py"


rule com_embedding:
    input:
        netfile=SIM_TWO_COM_NET,
    output:
        embfile=TWO_COM_EMB_FILE,
    params:
        model_name=lambda wildcards: wildcards.model_name,
        dim=lambda wildcards: wildcards.dim,
        window_length=lambda wildcards: wildcards.window_length,
        directed="undirected",
        num_walks=5,
    script:
        "workflow/embedding.py"


rule eval_auc_embedding:
    input:
        emb_files=TWO_COM_EMB_FILE_ALL,
    output:
        output_file=TWO_COM_AUC_FILE,
        output_sim_file=TWO_COM_SIM_FILE,
    script:
        "workflow/eval-community.py"


rule eval_embedding_kmeans:
    input:
        emb_files=TWO_COM_EMB_FILE_ALL,
    output:
        output_sim_file=RES_TWO_COM_KMEANS_FILE,
    script:
        "workflow/eval-community-kmeans.py"


rule plot_two_com_auc:
    input:
        input_file=TWO_COM_AUC_FILE,
    output:
        output_file=FIG_TWO_COM_AUC,
    script:
        "workflow/plot-two-com-auc.py"


#
# Multiple communities
#

MULTI_COM_NET_DIR = j(DATA_DIR, "networks", "multi_coms")
MULTI_COM_EMB_DIR = j(DATA_DIR, "embeddings", "milti_coms")
sim_net_params = {
    "n": [200, 500, 1000, 10000, 100000],
    "nc": [100],
    "cave": [50, 100, 200],
    "cdiff": [5, 10, 20, 30, 40], # cin - cout
    "sample": np.arange(10),
}
SIM_MULTI_COM_NET = j(
    MULTI_COM_NET_DIR, "net_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}.npz"
)
SIM_MULTI_COM_NET_ALL = expand(SIM_MULTI_COM_NET, **sim_net_params)

MULTI_COM_EMB_FILE_DIR = j(MULTI_COM_EMB_DIR, "embeddings")
emb_params_rw = {  # parameter for methods baesd on random walks
    "model_name": ["node2vec", "glove"],
    "window_length": [3, 5, 10],
    "dim": [1, 2, 8, 32, 128],
}
emb_params = {
    "model_name": [],
    #"model_name": ["leigenmap", "levy-word2vec", "adjspec", "modspec"],
    "window_length": [10],
    "dim": [1, 2, 8, 32, 128],
}
MULTI_COM_EMB_FILE = j(
    MULTI_COM_EMB_FILE_DIR,
    "embnet_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
)
MULTI_COM_EMB_FILE_ALL = expand(
    MULTI_COM_EMB_FILE, **sim_net_params, **emb_params
) + expand(MULTI_COM_EMB_FILE, **sim_net_params, **emb_params_rw)

rule generate_multi_com_net:
    params:
        cave=lambda wildcards: int(wildcards.cave),
        cdiff=lambda wildcards: int(wildcards.cdiff),
        n=lambda wildcards: int(wildcards.n),
        nc=lambda wildcards: int(wildcards.nc),
    output:
        output_file=SIM_MULTI_COM_NET,
    script:
        "workflow/generate-multi-com-net.py"

rule multi_com_embedding:
    input:
        netfile=SIM_MULTI_COM_NET,
    output:
        embfile=MULTI_COM_EMB_FILE,
    params:
        model_name=lambda wildcards: wildcards.model_name,
        dim=lambda wildcards: wildcards.dim,
        window_length=lambda wildcards: wildcards.window_length,
        directed="undirected",
        num_walks=5,
    script:
        "workflow/embedding.py"


rule _all:
    input:
        MULTI_COM_EMB_FILE_ALL
        #TWO_COM_EMB_FILE_ALL, #SIM_TWO_COM_NET_ALL
         #TWO_COM_SIM_FILE,RES_TWO_COM_KMEANS_FILE
         #TWO_COM_EMB_FILE_ALL


# rule some_data_processing:
# input:
# "data/some_data.csv"
# output:
# "data/derived/some_derived_data.csv"
# script:
# "workflow/scripts/process_some_data.py"
