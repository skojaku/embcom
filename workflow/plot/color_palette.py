# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-11 16:26:52
import seaborn as sns
import matplotlib.pyplot as plt


def get_model_order():
    return [
        "node2vec",
        "depthfirst-node2vec",
        "deepwalk",
        "line",
        "linearized-node2vec",
        "modspec",
        "leigenmap",
        "nonbacktracking",
        "bp",
        "infomap",
        "flatsbm",
        "non-backtracking-node2vec",
        "non-backtracking-deepwalk",
        "depthfirst-node2vec",
    ]


def get_model_names():
    return {
        "bp": "BP",
        "node2vec": "Neural node2vec",
        "linearized-node2vec": "Spectral node2vec",
        "modspec": "Modularity",
        "leigenmap": "L-EigenMap",
        "non-backtracking-node2vec": "Non-backtracking node2vec",
        "non-backtracking-deepwalk": "Non-backtracking DeepWalk",
        "depthfirst-node2vec": "Biased node2vec (p=10,q=0.1)",
        "deepwalk": "DeepWalk",
        "line": "LINE",
        "nonbacktracking": "Non-backtracking",
        "infomap": "Infomap",
        "flatsbm": "Flat SBM",
    }


def get_model_colors():
    cmap = sns.color_palette().as_hex()
    bcmap = sns.color_palette("bright").as_hex()
    mcmap = sns.color_palette("muted").as_hex()

    neural_emb_color = bcmap[1]
    spec_emb_color = bcmap[0]
    com_color = cmap[4]
    return {
        "node2vec": neural_emb_color,
        "deepwalk": sns.desaturate(neural_emb_color, 0.6),
        "line": sns.desaturate(neural_emb_color, 0.2),
        "linearized-node2vec": spec_emb_color,
        "modspec": sns.desaturate(spec_emb_color, 0.6),
        "leigenmap": sns.desaturate(spec_emb_color, 0.2),
        "bp": "k",
        "infomap": sns.desaturate(com_color, 1),
        "flatsbm": sns.desaturate(com_color, 0.6),
        "nonbacktracking": "white",
        "non-backtracking-node2vec": "red",
        "non-backtracking-deepwalk": "blue",
        "depthfirst-node2vec": sns.desaturate(neural_emb_color, 0.1),
    }


def get_model_edge_colors():
    return {
        "node2vec": "black",
        "deepwalk": "white",
        "line": "white",
        "linearized-node2vec": "black",
        "modspec": "white",
        "leigenmap": "white",
        "bp": "white",
        "nonbacktracking": "black",
        "non-backtracking-node2vec": "white",
        "non-backtracking-deepwalk": "white",
        "infomap": "white",
        "flatsbm": "white",
        "depthfirst-node2vec": "white",
    }


def get_model_linestyles():
    return {
        "node2vec": (1, 0),
        "deepwalk": (1, 1),
        "line": (2, 2),
        "linearized-node2vec": (1, 0),
        "modspec": (1, 1),
        "leigenmap": (2, 2),
        "nonbacktracking": (1, 0),
        "bp": (1, 0),
        "infomap": (1, 1),
        "flatsbm": (2, 2),
        "depthfirst-node2vec": (1, 3),
    }


def get_model_markers():
    return {
        "bp": "D",
        "node2vec": "s",
        "line": "s",
        "deepwalk": "s",
        "linearized-node2vec": "o",
        "modspec": "o",
        "leigenmap": "o",
        "nonbacktracking": "o",
        "non-backtracking-node2vec": "o",
        "depthfirst-node2vec": "o",
        "non-backtracking-deepwalk": "v",
        "infomap": "D",
        "flatsbm": "D",
    }


def get_model_groups():
    return {
        "bp": "community_detection",
        "node2vec": "neural",
        "line": "neural",
        "deepwalk": "neural",
        "linearized-node2vec": "spectral",
        "modspec": "spectral",
        "leigenmap": "spectral",
        "nonbacktracking": "spectral",
        "non-backtracking-node2vec": "neural",
        "depthfirst-node2vec": "neural",
        "non-backtracking-deepwalk": "neural",
        "infomap": "community_detection",
        "flatsbm": "community_detection",
    }
