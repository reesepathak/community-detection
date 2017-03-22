# SBM generator module
# EE376A
# Winter 2017

import numpy as np
import random

SBM_LIN = 0
SBM_LOG = 1

def gen_sbm(n, k, W):
    """
    gen_sbm: Generate a graph according to SBM(n, k, W)

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
        labels[i] = np.random.choice(range(len(k)), size = 1, 
                                     replace = True, p = k)

    # Iterate over all possible edges:
    edges = []
    for i in range(n):
        for j in range(i):
            val = np.random.random()
            if val <= W[labels[i]][labels[j]]:
                edges.append({i, j})
    return (labels, edges)

def gen_sym_sbm(n, k, a, b, regime=SBM_LIN):
    """
    gen_sym_sbm: Generate a graph according to the symmetric SBM model
                 in the linear or logarithmic regime.

    Parameters
    ----------
    n - int - the number of nodes.
    k - int - the number of clusters.
    a - float - Constant determining in-group edge probability.
    b - float - Constant determining out-group edge probability.
    regime - SBM_LIN or SBM_LOG
           - If Q = [a b b ... b
                     b a b ... b
                     b b a ... b
                     . . . ... .
                     b b b ... a] 
            then
           - In SBM_LIN, W = Q/n
           - In SBM_LOG, W = Q*ln(n)/n where ln is the natural logarithm.

    Returns
    -------
    labels - numpy vector describing the labels of the nodes 0, 1, ..., n-1
    edges - set of edges. Every edge is a set {i, j} indicating that an edge
            goes from i to j. Edges are not directed.
    """
    k_vec = np.ones(k)/k
    Q = np.diag((a-b)*np.ones(k)) + b*np.ones((k, k))
    if regime == SBM_LIN:
        return gen_sbm(n, k_vec, Q/n)
    elif regime == SBM_LOG:
        return gen_sbm(n, k_vec, Q*np.log(n)/n)
    else:
        raise ValueError("'regime' must be either 'SBM_LIN' or 'SBM_LOG'")


