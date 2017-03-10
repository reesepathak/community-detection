import random 
import numpy as np
from math import log, ceil

random.seed(1)
n = 1490 # this is the total number of nodes in the dataset, including 0-degree nodes

# load blog data
# TODO: come back and reason about the size of these nodes.
# in particular, the original paper says 1494 and we have 1490
# they also talk about filtering them
# AND the final community numbers don't add up properly
#
# TODO: should we ceil or floor r?
#
with open('data/polblogs.gml', 'r') as f:
    raw = f.readlines()

edges = set()
for r, line in enumerate(raw):
    if ' edge ' in line:
        v1 = int(raw[r+1].strip().split()[1]) - 1
        v2 = int(raw[r+2].strip().split()[1])
        if v1 == v2: continue
        v1, v2 = (v1 - 1, v2 - 1) # shift to 0-indexing

        e = (v1, v2) if v1 < v2 else (v2, v1)
        edges.add(e)
adj = np.zeros(n, dtype=object)
for i in range(n):
    adj[i] = set()

for (v1, v2) in edges:
    adj[v1].add(v2)
    adj[v2].add(v1)

d = 2.0 * len(edges) / len(nodes)
r_global = 3.0 / 4.0 * log(n) / log(d) # ASSUME r = r' 
c = 0.1

E = {e for e in edges if random.random() < c}
g_minus_e = edges - E

import random 
import numpy as np
from math import log
random.seed(1)
n = 1490 # this is the total number of nodes in the dataset, including 0-degree nodes
# load blog data
# TODO: come back and reason about the size of these nodes.
# in particular, the original paper says 1494 and we have 1490
# they also talk about filtering them
# AND the final community numbers don't add up properly
with open('data/polblogs.gml', 'r') as f:
    raw = f.readlines()
edges = set()
for r, line in enumerate(raw):
    if ' edge ' in line:
        v1 = int(raw[r+1].strip().split()[1]) - 1
        v2 = int(raw[r+2].strip().split()[1])
        if v1 == v2: continue
        v1, v2 = (v1 - 1, v2 - 1) # shift to 0-indexing
        e = (v1, v2) if v1 < v2 else (v2, v1)
        edges.add(e)
d = 2.0 * len(edges) / len(nodes)
global_r = 3.0 / 4.0 * log(n) / log(d) # ASSUME r = r' 
c = 0.1
E = {e for e in edges if random.random() < c}
g_minus_e = edges - E
adj = np.zeros(n, dtype=object)
for i in range(n):
    adj[i] = set()
for (v1, v2) in g_minus_e:
    adj[v1].add(v2)
    adj[v2].add(v1)

def nbors(r, v):
    """nbors[r, v] := his N_{r[G\E]}(v)"""
    depth = [0 for i in range(n)]
    visited = [False for i in range(n)]
    bfs_queue = [v]
    visited[v] = True
    while len(bfs_queue) > 0:
        front = bfs_queue.pop()
        for node in adj[front]:
            if not visited[node]:
                depth[node] = depth[front]+1
                visited[node] = True
                bfs_queue.append(node)
    ret = set()
    for node in range(n):
        if depth[node] == ceil(r): 
            ret.add(node)
    return ret
def nbor_mutuals(r, r_, v, v_):
    """nbor_mutuals[r, v, v2] := his N_{r, r' [E]}(v, v')"""
    nbor_rv = nbors(r, v)
    nbor_r_v_ = nbors(r_, v_)
    
    count = 0
    for v1 in nbor_rv:
        for v2 in nbor_r_v_:
            if (v1, v2) in E or (v2, v1) in E:
                count = count+1
    return count
def sign_stat(v, v_, r = global_r, r_ = global_r):
    """sign_stat[v, v2] := his I_{global_r, global_r, [E]}(v, v')"""
    x = nbor_mutuals(r, r_, v, v_)
    y = nbor_mutuals(r+1, r_, v, v_)
    z = nbor_mutuals(r+2, r_, v, v_)
    return x*z-y*y

# select 5 vertices randomly [v1, v2, ..., v5]
# for each pair (v_i, v_j)
#   compute sign_stat(r, r, v_i, v_j)
#
# try to find a partition such that sign_stat(v_i, v_j) > 0 
#   iff v_i and v_j are in the same community
#   if we can't, fail
#   if we can, pick a random node from each community to be our anchor
#
# for each other node w, assign it to maximize sign_stat(anchor(community), w)
v = random.sample(xrange(n), 5)
sign_stats = np.zeros((len(v), len(v)))
for i in range(len(v)):
    for j in range(len(v)):
        sign_stats[i][j] = sign_stat(v[i], v[j])

print "sign_stats", sign_stats

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
        for v in split2:
            if sign_stats[u][v] > 0:
                partition = False
    for u in split1:
        for v in split1:
            if sign_stats[u][v] <= 0:
                partition = False
    for u in split2:
        for v in split2:
            if sign_stats[u][v] <= 0:
                partition = False
    
    if partition:
        anchor1 = split1[0]
        anchor2 = split2[0]
        break
    # else:
    #     print "split1:", split1
    #     print "split2:", split2
if not partition:
    raise Exception("couldn't find partition")


