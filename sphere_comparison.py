# SBM test case
# EE 376A
# Winter 2017

import networkx as nx
import random
import numpy as np
import sbm

# Duplicates some functionality but works with networkx
def get_r_from_v(G, v, r):
    """
    Gets the set of vertices exactly r away from v in G

    Parameters
    ----------
    G - a networkx.Graph, expressed as an adjacency list
    v - a node in G
    r - a nonnegative int

    N_{r[G]}(v) in Abbe's notation.
    """
    # build adjacency list for speedup
    # out = []
    # bfs_queue = [v]
    # depth = np.zeros(max(Gadj) + 1, dtype=int) # Measured the distances to these nodes
    # visited = np.zeros(max(Gadj) + 1, dtype=bool) # Seen all the neighbors of these nodes
    # visited[v] = True
    # while len(bfs_queue) > 0:
    #     front = bfs_queue.pop()
    #     for nbor in Gadj[front]:
    #         if not visited[nbor]:
    #             visited[nbor] = True

    #             depth[nbor] = depth[front] + 1

    #             if depth[nbor] == r:
    #                 out.append(nbor)
    #             else:
    #                 bfs_queue.append(nbor)
    # return out
    lengths = nx.single_source_shortest_path_length(G, v, cutoff = r)
    Nr = [target for target in lengths if lengths[target] == r]
    return Nr


def count_sphere_crosses(G, v1, r1, v2, r2, edges):
    """
    Gets the number of edges (w, u) where w is r1 away from v1,
    u is r2 away from v2, and (w, u) is in edges.

    N_{r, r'[G/E]}(v, v') in Abbe's notation.
    """
    count = 0
    Nrv1 = get_r_from_v(G, v1, r1)
    Nrv2 = get_r_from_v(G, v2, r2)
    # print "size of Nrv1: ", len(Nrv1)
    # print "size of Nrv2: ", len(Nrv2)
    for node1 in Nrv1:
        for node2 in Nrv2:
            if (node1, node2) in edges or (node2, node1) in edges:
                count += 1
    # print "done."
    return count

def sphere_crossing_approx(n, a, b, k, c, r1, r2, delta):
    """
    Computes the approximation (given by Abbe) to count_sphere_crosses in terms of
    the model parameters.

    Parameters
    ----------
    a - in-group connectivity coeff.
    b - out-group connectivity coeff.
    k - number of communities
    c - prob that an edge is in E, the removed set.
    delta - 1 if v1, v2 are in the same community, 0 otherwise
    """
    r_exp = r1 + r2 + 1
    d = (1.0/k)*(a + (k-1)*b)
    leading_coeff = (1.0/n)*c*((1-c)**(r1+r2))
    inside = d**r_exp + (((a - b)/k)**r_exp )* (k*delta -1)
    return leading_coeff*inside

def sign_stat(G, v1, r1, v2, r2, edges):
    """
    Compute the sign-invariant statistic on v1 and v2.

    I_{r, r'[E]}(v, v') in Abbe's notation.
    """
    N0 = count_sphere_crosses(G, v1, r1, v2, r2, edges)
    N1 = count_sphere_crosses(G, v1, r1+1, v2, r2, edges)
    N2 = count_sphere_crosses(G, v1, r1+2, v2, r2, edges)
    print N0
    print N1
    print N2
    return N1*N1 - N0*N2

def gen_nx_sbm_component(n, k, a, b):
    # Generate (or read) SBM
    labels, edges = sbm.gen_sym_sbm(n, k, a, b)
    # Read into networkx:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.node[i]['label'] = labels[i]
    G.add_edges_from(edges)
    return G

def main():
    np.random.seed(1)
    # Graph parameters (hidden)
    n = 5000 # number of nodes
    k = 2     # number of components
    a = 15.4154       # in-group connectivity coeff.
    b = 1.3813     # out-group connectivity coeff.


    # Algorithm parameters (symmetric SBM)
    # Note: Algorithm needs (some) magical knowledge of k in the constant-degree regime.
    k_max = k
    c = 0.1            # Probability that an edge in G ends up in E
    num_anchors = int(np.ceil(k_max*np.log(4*k_max)))    # Number of nodes to pick

    ####################
    # Create a new SBM #
    ####################
    G = gen_nx_sbm_component(n, k, a, b)
    nx.write_adjlist(G, './sym_sbm.txt')
    with open("labels.txt", "w") as f:
        for v in G.nodes():
            f.write(str((G.node[v]['label'])) + "\n")

    ###################
    # Read an old SBM #
    ####################
    # with open('./sym_sbm.txt', 'r') as f:
    #     adj_lines = f.readlines()
    #     G = nx.parse_adjlist(adj_lines, nodetype=int)
    # with open("labels.txt", "r") as f:
    #     for v, label in enumerate(f.readlines()):
    #         G.node[v]['label']= int(label)


    # Get the largest connected component:
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)

    np.random.seed(2)
    # Remove random subset of edges.
    E = random.sample(G.edges(), np.int(len(G.edges())*c))
    G.remove_edges_from(E)

    # Pick nodes at random
    anchors = np.random.choice(G.nodes(), size = num_anchors, replace=False)

    # Convert G to adjacency list for speedup:
    Gadj = nx.to_dict_of_lists(G)

    # Compute average degree and r (r given in random formula by Abbe)
    avg_deg = np.mean([len(G.neighbors(v)) for v in G.nodes()]) # average node degree
    # r = int((0.75) * np.log(n)/np.log(avg_deg))
    # r = 3

    # Compute actual values of "estimates":
    lambda1 = (1.0/k)*(a + (k-1)*b)
    lambda2 = (1.0/k)*(a - b)
    print "a: ", a
    print "b: ", b
    print "num_nodes: ", len(G.nodes())
    print "num_edges [E]: ", len(E)
    print "num_edges [G\\E] ", len(G.edges())
    print "lambda1: ", lambda1
    print "lambda2: ", lambda2
    print "SNR: ", float((a - b)**2)/(k*(a + (k-1)*b))
    print "lambda1^7 < lambda2^8? ", bool(lambda1**7 < lambda2**8)
    print "4*lambda1^3 < lambda2^4? ", bool(4*lambda1**3 < lambda2**4)
    print "average degree (should be close to lambda1): ", avg_deg
    # print "SNR: ", lambda2**2/lambda1
    # P = np.diag((1.0/k)*np.ones(k))
    # Q = np.diag((a-b)*np.ones(k)) + b*np.ones((k, k))
    # e_vals, e_vecs = np.linalg.eig(np.dot(P, Q))
    # print e_vals

    # Expected number of r-away neighbors in and out of same community
    r_max = int((0.75) * np.log(n)/np.log(avg_deg))
    for r in range(1, r_max +1):
        Nr_intra_expect = (1.0/k)*(lambda1**r+(k-1)*lambda2**r)
        Nr_extra_expect = (1.0/k)*(lambda1**r - lambda2**r)
        print "r: ", r
        print "Expected: " + str(Nr_intra_expect) + ", " + str(Nr_extra_expect)
        Nr_intra_actual_vec = []
        Nr_extra_actual_vec = []
        for v in G.nodes():
            # Nr = get_r_from_v(Gadj, v, r)
            lengths = nx.single_source_shortest_path_length(G, v, cutoff = r)
            Nr = [target for target in lengths if lengths[target] == r]
            Nr_intra_actual = 0
            Nr_extra_actual = 0
            for node in Nr:
                if G.node[node]['label'] == G.node[v]['label']:
                    Nr_intra_actual += 1
                else:
                    Nr_extra_actual += 1
            Nr_intra_actual_vec.append(Nr_intra_actual)
            Nr_extra_actual_vec.append(Nr_extra_actual)

            # print "Observed: " + str(Nr_intra_actual) + ", " + str(Nr_extra_actual)
            # print "percent error: " + \
                  # str(100*np.abs(Nr_intra_actual - Nr_intra_expect)/Nr_intra_expect) + ", " + \
                  # str(100*np.abs(Nr_extra_actual - Nr_extra_expect)/Nr_extra_expect)
        print "mean_intra_actual: ", np.mean(Nr_intra_actual_vec), " standard_dev: ", np.std(Nr_intra_actual_vec)
        print "mean_extra_actual: ", np.mean(Nr_extra_actual_vec), " standard_dev: ", np.std(Nr_extra_actual_vec)


    sign_stats = np.zeros((len(anchors), len(anchors)))
    for i, v1 in enumerate(anchors):
        for j, v2 in enumerate(anchors):
            sign_stats[i][j] = sign_stat(G, v1, 1, v2, r_max, E)
    print sign_stats
    for v in anchors:
        print str(v) + " : " + str(G.node[v]['label'])

    # Test: What happens for two nodes if we hold r+r' constant, but vary r:
    v1 = anchors[0]
    v2 = anchors[3]
    crosses = np.zeros(r_max)
    crosses_approx = np.zeros(r_max)
    for s1 in range(1, r_max +1):
        s2 = r_max - s1
        crosses[s1-1] = count_sphere_crosses(G, v1, s1, v2, s2, E)
        delta = int(G.node[v1]['label'] == G.node[v2]['label'])
        crosses_approx[s1-1] = sphere_crossing_approx(n, a, b, k, c, s1, r_max, delta)
    print "v1: ", G.node[v1]['label']
    print "v2: ", G.node[v2]['label']
    print crosses
    print crosses_approx


if __name__ == "__main__":
    main()




