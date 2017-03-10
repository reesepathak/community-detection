import random 
from math import log

random.seed(1)

# load blog data
# TODO: come back and reason about the size of these nodes.
# in particular, the original paper says 1494 and we have 1490
# they also talk about filtering them
# AND the final community numbers don't add up properly
with open('data/polblogs.gml', 'r') as f:
    raw = f.readlines()

edges = set()
nodes = set()
for r, line in enumerate(raw):
    if ' node ' in line:
        v = int(raw[r+1].strip().split()[1])
        nodes.add(v)

    if ' edge ' in line:
        v1 = int(raw[r+1].strip().split()[1])
        v2 = int(raw[r+2].strip().split()[1])
        if v1 == v2: continue

        e = (v1, v2) if v1 < v2 else (v2, v1)
        edges.add(e)
adj = {v : set() for v in nodes}
for (v1, v2) in edges:
    adj[v1].add(v2)
    adj[v2].add(v1)

n = len(nodes)
d = 2.0 * len(edges) / len(nodes)
r_global = 3.0 / 4.0 * log(n) / log(d) # ASSUME r = r' 
c = 0.1

E = {e for e in edges if random.random() < c}
g_minus_e = edges - E

def nbors(r, v):
    """nbors[G, r, v] := his N_{r[G]}(v)"""
    pass

def nbor_mutuals(r, v, v2):
    """nbor_mutuals[G\E, r, r', v, v'] := his N_{r, r' [E]}(v, v')"""
    pass

def sign_stat(v, v2):
    """sign_stat[G\E, r, r', v, v'] := his I_{r, r', [E]}(v, v')"""
    pass

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


