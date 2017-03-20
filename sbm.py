# SBM generator module
# EE376A
# Winter 2017

import numpy as np
import random

def gen_SBM(n, k, W):
    """
    gen_SBM: Generate a graph according to SBM(n, k, W)

    Parameters
    ----------
    n - int denoting the number of nodes.
    k - numpy array containing numbers 0 < k[i] < 1 and sum(k) = 1
        representing the probability that a random node is in community k[i]
    W - square numpy array containing numbers 0 < W[i][j] < 1
        W[i][j] is the probability that given nodes in communites i and j, there
        is an edge connecting them.

    Returns
    -------
    labels - numpy vector describing the labels of the nodes 0, 1, ..., n-1
    edges - set of edges. Every edge is a set {i, j} indicating that an edge
            goes from i to j. Edges are not directed.
    """
    # Assign labels:
    labels = np.zeros(n, dtype = int)
    for i in range(n):
        labels[i] = np.random.choice(n, size = 1, replace = True, p = k)

    # Iterate over all possible edges:
    edges = []
    for i in range(n):
        for j in range(i):
            val = np.random.random()
            if val <= W[labels[i]][labels[j]]:
                edges.append({i, j})
    return (labels, edges)


