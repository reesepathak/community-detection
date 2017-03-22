# SDP Exact Recovery
# arXiv:1405.3267
# EE376A Winter 2017

import cvxpy as cvx
import numpy as np
import networkx as nx
import sbm

def gen_nx_sbm_component(n, k, a, b, regime=sbm.SBM_LIN):
    # Generate (or read) SBM
    labels, edges = sbm.gen_sym_sbm(n, k, a, b, regime = regime)
    # Read into networkx:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.node[i]['label'] = labels[i]
    G.add_edges_from(edges)
    return G

def main():
    # Generate/Read SBM
    np.random.seed(1)
    # Graph parameters (hidden)
    n = 100 # number of nodes
    k = 2     # number of components
    a = 15.4154       # in-group connectivity coeff.
    b = 1.3813     # out-group connectivity coeff.

    assert (a - b)**2 > 8*(a+b) + (8.0/3)*(a-b)
    ####################
    # Create a new SBM #
    ####################
    print "Generating sbm..."
    G = gen_nx_sbm_component(n, k, a, b, regime = sbm.SBM_LOG)
    nx.write_adjlist(G, './sym_sbm_sdp.txt')
    with open("labels_sdp.txt", "w") as f:
        for v in G.nodes():
            f.write(str((G.node[v]['label'])) + "\n")
    print "num_nodes: ", len(G.nodes())
    print "num_edges: ", len(G.edges())
    ###################
    # Read an old SBM #
    ####################
    # print "Reading sbm..."
    # with open('./sym_sbm_sdp.txt', 'r') as f:
    #     adj_lines = f.readlines()
    #     G = nx.parse_adjlist(adj_lines, nodetype=int)
    # with open("labels_sdp.txt", "r") as f:
    #     for v, label in enumerate(f.readlines()):
    #         G.node[v]['label']= int(label)

    #########################
    # Construct Edge matrix #
    #########################
    print "Constructing SDP..."
    B = np.diag(np.ones(n)) - np.ones((n, n)) # Zero diagonal, -1 off diagonal.
    print "\tSetting edges..."
    for v1, v2 in G.edges():
        B[v1][v2] = 1
        B[v2][v1] = 1

    ######################
    # Set up CVX Problem #
    ######################
    print "\tSetting up constraints..."
    X = cvx.Semidef(n)
    objective = cvx.trace(B*X)
    constraints = []
    for i in range(n):
        constraints += [X[i,i] == 1]
    constraints += [X >> 0]
    prob = cvx.Problem(cvx.Maximize(objective), constraints)

    print "Solving SDP..."
    prob.solve(solver = "SCS", verbose = True)
    print "Done."

    ## By paper, X is the outer product of some vector.
    e_vals, e_vecs = np.linalg.eigs(X)
    print e_vecs[0]

if __name__ == "__main__":
    main()
