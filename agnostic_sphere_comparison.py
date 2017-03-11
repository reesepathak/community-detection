import random 
import numpy as np
from math import log, ceil

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

# TODO still mysterious what r' should be
global_r = ceil(3.0 / 4.0 * log(len(nodes)) / log(d)) 
global_r_ = 1

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

# select 5 vertices randomly and compute pairwise sign statistics
v = random.sample(nodes, 5)
sign_stats = np.zeros((len(v), len(v)))
for i in range(len(v)):
    for j in range(len(v)):
        sign_stats[i][j] = sign_stat(v[i], v[j])

print "v:", v
print "sign_stats:\n", sign_stats.astype(int)

# check if there exists a consistent partition of these vertices
#   i.e. sign_stat[v_i, v_j] > 0 iff v_i and v_j are in the same community
for i in range(2**len(v)):
    split1 = []
    split2 = []
    j = i
    ctr = 0
    while j:
        if j%2:
            split1.append(ctr)
        else:
            split2.append(ctr)
        ctr = ctr+1
        j >>= 1
    if not len(split1) or not len(split2):
        continue
    partition = True
    for u in split1:
        for w in split2:
            if sign_stats[u][w] > 0:
                partition = False
    for u in split1:
        for w in split1:
            if sign_stats[u][w] <= 0:
                partition = False
    for u in split2:
        for w in split2:
            if sign_stats[u][w] <= 0:
                partition = False
    
    if partition:
        anchor1 = split1[0]
        anchor2 = split2[0]
        break

if not partition:
    raise Exception("couldn't find partition")

# TODO: for each other node w, assign it to maximize sign_stat(anchor(community), w)



