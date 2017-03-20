# SBM test case
# EE 376A
# Winter 2017

import networkx as nx
import numpy as np
import sbm

# Duplicates some functionality but works with networkx
def get_r_from_v(G, v, r):
    """
    Gets the set of vertices exactly r away from v in G

    Parameters
    ----------
    G - a networkx.Graph
    v - a node in G
    r - a nonnegative int


    N_{r[G]}(v) in Abbe's notation.
    """
    out = {}
    bfs_queue = [v]
    dist = {v : 0} # Measured the distances to these nodes
    visited = {} # Seen all the neighbors of these nodes
    while len(bfs_queue) > 0:
        curr = bfs_queue.pop()
        if dist[curr] == r:
            out.add(curr)
            continue
        for node in G.neighbors(curr):
            if node not in visited:
                dist[node] = dist[curr] + 1
                bfs_queue.append(node)
            visited.add(curr)
    return out

def count_sphere_crosses(G, v1, r1, v2, r2, edges):
    """
    Gets the number of edges (w, u) where w is r1 away from v1,
    u is r2 away from v2, and (w, u) is in edges.

    N_{r, r'[G/E]}(v, v') in Abbe's notation.
    """
    count = 0
    Nrv1 = get_r_from_v(G, v1, r1)
    Nrv2 = get_r_from_v(G, v2, r2)
    for node1 in Nrv1:
        for node2 in Nrv2:
            if (node1, node2) in edges or (node2, node1) in edges:
                count += 1
    return count

if __name__ == "__main__":
    np.random.seed(1)
    # Graph parameters
    n = 1500 # number of nodes
    k = 2    # number of components
    a = 10   # in-group connectivity coeff.
    b = 2    # out-group connectivity coeff.

    # Algorithm parameters (symmetric SBM)
    d = (a + (k-1)*b)  # Average degree
    c = 1/10           # Probability that an edge in G ends up in E
    num_anchors = 4    # number of nodes to pick

    # Generate SBM
    labels, edges = sbm.gen_sym_sbm(n, k, a, b)

    # Read into networkx:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.node[i]['label'] = labels[i]
    G.add_edges_from(edges)

    # Get the largest connected component:
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)

    # Pick nodes at random
    anchors = np.random.choice(G.nodes(), size = num_anchors)




