import logging
import os

import fastnode2vec
import numpy as np
import pandas as pd
from scipy import sparse

import embcom

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

#
# Input
#
netfile = snakemake.input["netfile"]
nodefile = snakemake.input["nodefile"] if "nodefile" in snakemake.input.keys() else None
dim = int(snakemake.params["dim"])
window_length = int(snakemake.params["window_length"])
model_name = snakemake.params["model_name"]
directed = snakemake.params["directed"] == "directed"
noselfloop = (
    snakemake.params["noselfloop"] == "True"
    if "noselfloop" in snakemake.params.keys()
    else False
)
num_walks = (
    int(snakemake.params["num_walks"]) if "num_walks" in snakemake.params.keys() else 1
)
embfile = snakemake.output["embfile"]

net = sparse.load_npz(netfile)

if nodefile is not None:
    node_table = pd.read_csv(nodefile)

if directed is False:
    net = net + net.T

if noselfloop:
    net.setdiag(0)
    logger.debug("Remove selfloops")

#
# Embedding models
#
if model_name == "levy-word2vec":
    model = embcom.embeddings.LevyWord2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "node2vec":
    model = fastnode2vec.Node2Vec(window_length=window_length, num_walks=num_walks)
elif model_name == "depthfirst-node2vec":
    model = fastnode2vec.Node2Vec(
        window_length=window_length, num_walks=num_walks, p=10, q=0.1
    )
elif model_name == "node2vec-qhalf":
    model = fastnode2vec.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks, q=0.5
    )
elif model_name == "node2vec-pagerank":
    model = embcom.embeddings.Node2Vec(
        window_length=window_length,
        restart_prob=0.1,
        num_walks=num_walks,
        random_teleport=True,
    )
elif model_name == "node2vec-qdouble":
    model = fastnode2vec.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks, q=2
    )
elif model_name == "deepwalk":
    model = embcom.embeddings.DeepWalk(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "glove":
    model = embcom.embeddings.Glove(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "leigenmap":
    model = embcom.embeddings.LaplacianEigenMap()
elif model_name == "adjspec":
    model = embcom.embeddings.AdjacencySpectralEmbedding()
elif model_name == "modspec":
    model = embcom.embeddings.ModularitySpectralEmbedding()
elif model_name == "nonbacktracking":
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()

#
# Embedding
#
model.fit(sparse.csr_matrix(net))
emb = model.transform(dim=dim)
out_emb = model.transform(dim=dim, return_out_vector=True)

#
# Save
#
np.savez(
    embfile,
    emb=emb,
    out_emb=out_emb,
    window_length=window_length,
    dim=dim,
    directed=directed,
    model_name=model_name,
)
