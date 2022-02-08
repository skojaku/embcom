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
netfile = snakemake.input["net_file"]
embfile = snakemake.output["output_file"]
params = snakemake.params["parameters"]
dim = int(params["dim"])
window_length = int(params["window_length"])
model_name = params["model_name"]
num_walks = 20

net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0 + 1

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

#
# Save
#
np.savez(
    embfile, emb=emb, window_length=window_length, dim=dim, model_name=model_name,
)
