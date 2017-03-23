# sbm.jl
# EE376A Winter 2017
# SBM generator module
# EE376A
# Winter 2017

SBM_LIN = 0
SBM_LOG = 1

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
function gen_sbm(n, k, W)
    # Assign labels:
    labels = zeros(Int64, n)
    for i in 1:n
        labels[i] = sample([1, 2], k)
    end
    # Iterate over all possible edges:
    for i in 1:n
        for j in 1:i
            val = rand()
            if val <= W[labels[i], labels[j]]
                push!(edges, (i, j))
            end
        end
    end
    return (labels, edges)
end


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
function gen_sym_sbm(n, k, a, b, regime=SBM_LIN)
    k_vec = ones(k)/k
    Q = diagm((a-b)*ones(k)) + b*ones((k, k))
    if regime == SBM_LIN
        return gen_sbm(n, k_vec, Q/n)
    elseif regime == SBM_LOG
        return gen_sbm(n, k_vec, Q*log(n)/n)
    else
        throw(TypeError("'regime' must be either 'SBM_LIN' or 'SBM_LOG'"))
    end
end

function to_lightgraph(labels, edges)
    labels = labels - 1 
    edges = [(i-1, j-1) for (i,j) in edges]
    G = Graph()
    add_vertices!(G, 1:n)
    for (i, j) in edges
        add_edge!(G, i, j)
    end
    return G, labels
end

