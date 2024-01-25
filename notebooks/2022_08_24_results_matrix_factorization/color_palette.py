import seaborn as sns
import matplotlib.pyplot as plt


def get_model_order():
    return [
        "non-backtracking-node2vec",
        "non-backtracking-deepwalk",
        "nonbacktracking",
        "node2vec",
        "depthfirst-node2vec",
        "deepwalk",
        "line",
        "linearized-node2vec",
        "modspec",
        "leigenmap",
        "infomap",
        "flatsbm",
    ]


def get_model_colors():
    cmap = sns.color_palette(desat=0.9).as_hex()

    c1 = sns.light_palette(cmap[0], n_colors=6, reverse=True)
    c2 = sns.light_palette(cmap[1], n_colors=6, reverse=True)
    c3 = sns.light_palette("#2d2d2d", n_colors=3, reverse=True)

    return {
        "non-backtracking-node2vec": "red",
        "non-backtracking-deepwalk": "blue",
        "node2vec": c1[0],
        "depthfirst-node2vec": c1[2],
        "deepwalk": c1[3],
        "line": c1[4],
        "linearized-node2vec": c2[0],
        "modspec": c2[1],
        "nonbacktracking": c2[2],
        "leigenmap": c2[3],
        "infomap": c3[0],
        "flatsbm": c3[1],
    }


def get_model_linestyles():
    return {
        "non-backtracking-node2vec": (0, 0),
        "non-backtracking-deepwalk": (0, 0),
        "node2vec": (1, 1),
        "depthfirst-node2vec": (1, 3),
        "deepwalk": (1, 4),
        "line": (2, 4),
        "linearized-node2vec": (0, 0),
        "modspec": (1, 1),
        "nonbacktracking": (0, 0),
        "leigenmap": (1, 1),
        "infomap": (0, 0),
        "flatsbm": (1, 1),
    }


def get_model_markers():
    return {
        "non-backtracking-node2vec": "s",
        "non-backtracking-deepwalk": "s",
        "node2vec": "o",
        "depthfirst-node2vec": "D",
        "deepwalk": "v",
        "line": "^",
        "linearized-node2vec": "s",
        "modspec": "o",
        "nonbacktracking": "D",
        "leigenmap": "v",
        "infomap": "s",
        "flatsbm": "o",
    }


def get_model_names():
    return {
        "non-backtracking-node2vec": "Non-backtracking node2vec",
        "non-backtracking-deepwalk": "Non-backtracking DeepWalk",
        "depthfirst-node2vec": "Biased node2vec (p=10,q=0.1)",
        "node2vec": "Unbiased node2vec (p=q=1)",
        "linearized-node2vec": "Linearized node2vec",
        "deepwalk": "DeepWalk",
        "line": "LINE",
        "modspec": "Modularity embedding",
        "nonbacktracking": "Non-backtracking embedding",
        "leigenmap": "Laplacian EigenMap",
        "infomap": "Infomap",
        "flatsbm": "Flat SBM",
    }
