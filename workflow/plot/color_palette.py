# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-10 20:57:56
import seaborn as sns
import matplotlib.pyplot as plt


def get_model_order():
    return [
        "bp",
        "node2vec",
        "depthfirst-node2vec",
        "deepwalk",
        "line",
        "nonbacktracking",
        "modspec",
        "leigenmap",
        "infomap",
        "flatsbm",
        "linearized-node2vec",
        "non-backtracking-node2vec",
        "non-backtracking-deepwalk",
    ]


def get_model_colors():
    cmap = sns.color_palette().as_hex()

    c1 = sns.light_palette(cmap[1], n_colors=6, reverse=True)
    c2 = sns.light_palette(cmap[0], n_colors=6, reverse=True)
    c3 = sns.light_palette(cmap[4], n_colors=6, reverse=True)

    # c1 = sns.light_palette(cmap[1], n_colors=6, reverse=True)
    # c2 = sns.light_palette(cmap[0], n_colors=6, reverse=True)
    # c3 = sns.light_palette(cmap[4], n_colors=6, reverse=True)
    return {
        "bp": "black",
        "non-backtracking-node2vec": "red",
        "non-backtracking-deepwalk": "blue",
        "node2vec": "red",
        "deepwalk": c1[0],
        "line": c1[2],
        "nonbacktracking": "blue",
        "linearized-node2vec": c2[0],
        "modspec": c2[2],
        "leigenmap": c2[3],
        "infomap": c3[0],
        "flatsbm": c3[2],
        "depthfirst-node2vec": c1[2],
    }


def get_model_linestyles():
    return {
        "bp": (0, 0),
        "non-backtracking-node2vec": (0, 0),
        "non-backtracking-deepwalk": (0, 0),
        "node2vec": (1, 1),
        "depthfirst-node2vec": (1, 3),
        "deepwalk": (1, 4),
        "line": (2, 4),
        "linearized-node2vec": (0, 0),
        "modspec": (1, 1),
        "nonbacktracking": (2, 2),
        "leigenmap": (3, 3),
        "infomap": (0, 0),
        "flatsbm": (1, 1),
    }


def get_model_markers():
    return {
        "bp": "*",
        "non-backtracking-node2vec": "o",
        "depthfirst-node2vec": "o",
        "node2vec": "o",
        "non-backtracking-deepwalk": "v",
        "deepwalk": "v",
        "line": "^",
        "linearized-node2vec": "o",
        "nonbacktracking": "D",
        "modspec": "v",
        "leigenmap": "v",
        "infomap": "s",
        "flatsbm": "s",
    }


def get_model_names():
    return {
        "bp": "Belief propagation",
        "non-backtracking-node2vec": "Non-backtracking node2vec",
        "non-backtracking-deepwalk": "Non-backtracking DeepWalk",
        "depthfirst-node2vec": "Biased node2vec (p=10,q=0.1)",
        "node2vec": "node2vec (p=q=1)",
        "linearized-node2vec": "Linearized node2vec",
        "deepwalk": "DeepWalk",
        "line": "LINE",
        "modspec": "Modularity embedding",
        "nonbacktracking": "Non-backtracking",
        "leigenmap": "Laplacian EigenMap",
        "infomap": "Infomap",
        "flatsbm": "Flat SBM",
    }
