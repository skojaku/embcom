import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import embcom


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_mod_spectral(self):
        model = embcom.embeddings.ModularitySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_adj_spectral(self):
        model = embcom.embeddings.AdjacencySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_leigenmap(self):
        model = embcom.embeddings.LaplacianEigenMap()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_node2vec(self):
        model = embcom.embeddings.Node2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_glove(self):
        model = embcom.embeddings.Glove()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_deepwalk(self):
        model = embcom.embeddings.DeepWalk()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_levy_walk(self):
        model = embcom.embeddings.LevyWord2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_nonbacktracking(self):
        model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_node2vec_matrix_factorization(self):
        model = embcom.embeddings.Node2VecMatrixFactorization()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_highorder_modularity_spec_embedding(self):
        model = embcom.embeddings.HighOrderModularitySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)

    def test_normalized_trans_matrix_spec_embedding(self):
        model = embcom.embeddings.NormalizedTransitionMatrixSpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=32)


if __name__ == "__main__":
    unittest.main()
