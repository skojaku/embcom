# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-21 17:12:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-21 17:13:52

from scipy.stats import binom
import random
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt


def sbm_model(N, pin, pout):

    split = int(N / 2)

    memberships = {}

    clusters = {}
    c = 0
    clusters[c] = []
    for n in range(0, split):
        clusters[c].append(n)
        memberships[n] = c
    c = 1
    clusters[c] = []
    for n in range(split, N):
        clusters[c].append(n)
        memberships[n] = c

    bond = {}
    for n in range(0, N):
        bond[n] = set()

    list_of_clusters = list(clusters.keys())

    for i in range(0, len(list_of_clusters)):

        c = clusters[i]

        S = int(len(c) * (len(c) - 1) / 2)
        Ein = binom.rvs(S, pin, size=1)[0]
        e = 0
        while e < Ein:
            s = random.sample(c, 2)
            n = s[0]
            m = s[1]
            if n not in bond[m]:
                bond[n].add(m)
                bond[m].add(n)
                e = e + 1

    for i in range(0, len(list_of_clusters) - 1):

        c = clusters[i]

        for j in range(i + 1, len(list_of_clusters)):

            q = clusters[j]

            S = int(len(c) * len(q))
            Eout = binom.rvs(S, pout, size=1)[0]

            e = 0
            while e < Eout:
                n = random.sample(c, 1)[0]
                m = random.sample(q, 1)[0]

                if n not in bond[m]:
                    bond[n].add(m)
                    bond[m].add(n)
                    e = e + 1

    return bond, memberships


def spectral_modularity(bond):
    M = 0.0
    degree = {}
    for n in bond:
        degree[n] = float(len(bond[n]))
        M += float(len(bond[n]))

    ##########
    vec = {}
    mu = 0.0
    for n in bond:
        vec[n] = 0.5 - random.random()
        mu += vec[n] * vec[n]
    for n in bond:
        vec[n] /= np.sqrt(mu)
    ###########
    err = 1.0
    #     print (mu, err)
    iteration = 0
    while err > 1.0 / M and iteration < 100:
        vec, mu, err = spectral_modularity_single_iteration(bond, degree, M, vec)
        iteration += 1
        # print (mu, err)

    ###########
    return mu, vec


################################


def spectral_modularity_single_iteration(bond, degree, M, vec):

    ###################

    new_vec = {}
    summ = 0.0
    for n in bond:
        summ += vec[n] * degree[n] / M

    for n in bond:
        new_vec[n] = 0.0
        for m in bond[n]:
            new_vec[n] += vec[m]

    for n in bond:
        new_vec[n] -= degree[n] * summ

    mu = 0.0
    for n in bond:
        mu += new_vec[n] * new_vec[n]

    sign = 1
    if new_vec[0] < 0:
        sign = -1

    err = 0.0
    for n in bond:
        new_vec[n] = sign * new_vec[n] / np.sqrt(mu)
        tmp = np.abs(vec[n] - new_vec[n])
        if tmp > err:
            err = tmp

    return new_vec, mu, err


def nmi_inferred_clusters(vec, memberships):

    x, y = [], []
    for n in vec:
        if vec[n] > 0:
            x.append(1)
        else:
            x.append(0)

        y.append(memberships[n])

    return normalized_mutual_info_score(x, y)
