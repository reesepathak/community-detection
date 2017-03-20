Pkg.add("LightGraphs")
using LightGraphs

function graph_from_file(fname)
    """
    Reads in a graph from file
    Args:
        fname: path to file
    Returns: LightGraph object
    """
    orig_graph = loadgraph(fname, :gml)
    # obtain list of weakly connected subgraphs
    list_of_WCCs = weakly_connected_components(orig_graph)
    # obtain index of largest such weakly connected component
    ind_max = indmax(map(x->size(x)[1], list_of_WCCs))
    G, node_list = induced_subgraph(orig_graph, node_list)
    return G, node_list
end
