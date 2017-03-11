import random 
import numpy as np
from math import log, ceil
from itertools import combinations

import networkx as nx

random.seed(1)

# load blog data
with open('data/polblogs.gml', 'r') as f:
    data = f.readlines()

raw_edges = set()
labels = {}
for row, line in enumerate(data):
    if ' node ' in line:
        v = int(data[row+1].strip().split()[1])
        v -= 1 # shift to 0-indexing
        labels[v] = int(data[row+3].strip().split()[1])

    if ' edge ' in line:
        v1 = int(data[row+1].strip().split()[1])
        v2 = int(data[row+2].strip().split()[1])
        if v1 == v2: continue
        v1, v2 = (v1 - 1, v2 - 1) # shift to 0-indexing
        e = (v1, v2) if v1 < v2 else (v2, v1)
        raw_edges.add(e)

# use networkx to find the largest component
G = nx.Graph()
G.add_edges_from(raw_edges)
Gc = max(nx.connected_component_subgraphs(G), key=len)
edges = set(Gc.edges())
nodes = Gc.nodes()
max_node = max(nodes)

d = 2.0 * len(edges) / len(nodes)

global_r = ceil(3.0 / 4.0 * log(len(nodes)) / log(d)) 
global_r_ = 1 # TODO still mysterious what r' should be

c = 0.1
E = {e for e in edges if random.random() < c}
g_minus_e = edges - E

adj = {}
for v in nodes:
    adj[v] = set()
for (v1, v2) in g_minus_e:
    adj[v1].add(v2)
    adj[v2].add(v1)

def nbors(r, v):
    """aka his N_{r[G\E]}(v)"""
    depth = np.zeros(max_node + 1, dtype=int)
    visited = np.zeros(max_node + 1, dtype=bool)
    bfs_queue = [v]
    visited[v] = True

    ok = []
    while len(bfs_queue) > 0:
        front = bfs_queue.pop()
        for nbor in adj[front]:
            if not visited[nbor]:
                visited[nbor] = True

                depth[nbor] = depth[front] + 1

                if depth[nbor] == r:
                    ok.append(nbor)
                else:
                    bfs_queue.append(nbor)
    return ok

def mutual_nbor_count(r, r_, v, v_):
    """aka N_{r, r' [E]}(v, v_)"""
    nbor_rv = nbors(r, v)
    nbor_r_v_ = nbors(r_, v_)
    
    count = 0
    for v1 in nbor_rv:
        for v2 in nbor_r_v_:
            if (v1, v2) in E or (v2, v1) in E:
                count +=1
    return count

def sign_stat(v, v_, r = global_r, r_ = global_r_):
    """aka I_{global_r, global_r, [E]}(v, v_)"""
    x = mutual_nbor_count(r, r_, v, v_)
    y = mutual_nbor_count(r+1, r_, v, v_)
    z = mutual_nbor_count(r+2, r_, v, v_)
    return x*z - y*y

def partition(vertices, stats):
    """ Returns a consistent partition of the vertices into two groups:
        i.e. stats[i, j] > 0 iff i and j are in the same group
        Throws an exception if none exists.
    """
    assert len(vertices) == stats.shape[0] == stats.shape[1]
    vertices = np.array(vertices)
    signs = (stats > 0)
    indices = np.arange(len(vertices))

    # check symmetry
    for i in indices:
        for j in indices:
            if signs[i][j] != signs[j][i]:
                raise ValueError("Sign of (%s, %s) = %s is inconsistent with (%s, %s) = %s" 
                    % (vertices[i], vertices[j], stats[i, j],
                        vertices[j], vertices[i], stats[j, i]))
    A = [indices[0]]
    B = []
    
    for i in indices[1:]:
        if np.all(signs[i,A]) and not np.any(signs[i,B]):
            A.append(i)
        elif np.all(signs[i,B]) and not np.any(signs[i,A]):
            B.append(i)
        else:
            raise ValueError("No consistent way to assign %s into %s or %s" % 
                (vertices[i], vertices[A], vertices[B]))

    if len(B) == 0:
        raise ValueError("All of the nodes are in the same partition")

    return (vertices[A], vertices[B])

# select 5 vertices randomly and compute pairwise sign statistics
vs = random.sample(nodes, 5)
sign_stats = np.zeros((len(vs), len(vs)))
for i in range(len(vs)):
    for j in range(len(vs)):
        sign_stats[i][j] = sign_stat(vs[i], vs[j])

print "vs:", vs
print "int(sign_stats):\n", sign_stats.astype(int)

C1, C2 = partition(vs, sign_stats)
print "C1, C2:", C1, C2

# TODO: for each other node w, assign it to maximize sign_stat(anchor(community), w)



