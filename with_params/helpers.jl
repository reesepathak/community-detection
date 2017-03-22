Pkg.add("LightGraphs")
using LightGraphs, StatsBase

function findNeighborhood(graph, source, size)
    selfset = Set{Int64}([source])
    balls, seen_nodes, dist = [selfset], selfset, 0
    distances = Dict()
    while length(seen_nodes) < size
        dist += 1
        curr_neighbors = Set(neighborhood(graph, source, dist))
        setdiff!(curr_neighbors, seen_nodes)
        for neighbor in curr_neighbors
            push!(distances, neighbor=>dist)
        end
        push!(balls, curr_neighbors)
        union!(seen_nodes, curr_neighbors)
    end
    push!(distances, source=>0)
    return (balls, distances, dist)
end

# Hyperparmater is 8 here
function vertexCompare(graph, u, v)
    if (degree(graph, u) * degree(graph, v)) == 0
        return nothing 
    end
    v_neighbors = findNeighborhood(graph, v, 8)
    u_neighbors = findNeighborhood(graph, u, 8)
    return checkOverlap(graph, u_neighbors, v_neighbors)
end

function checkOverlap(graph, u_neighbors, v_neighbors)
    ball_u, dist_u, dmax_u = u_neighbors
    ball_v, dist_v, dmax_v = v_neighbors
    outer_ball_u, outer_ball_v = ball_u[dmax_u + 1], ball_v[dmax_v + 1]
    edge_count_u = sum([length(neighbors(graph, tmp)) - 1 for tmp in outer_ball_u])
    edge_count_v = sum([length(neighbors(graph, tmp)) - 1 for tmp in outer_ball_v])
    candidates = edge_count_v * edge_count_u
    if (candidates == 0)
        return nothing
    end
    count = 0
    for vert1 in outer_ball_v
        for vert2 in neighbors(graph, vert1)
            if ((vert2 ∉ keys(dist_v)) || dist_v[vert2] > dmax_v) && ((vert2 ∉ keys(dist_u)) || dist_u[vert2] > dmax_u)
                for vert3 in neighbors(graph, vert2)
                    if (vert3 ∈ keys(dist_u) && dist_u[vert3] == dmax_u && vert3 != vert1)
                        count += 1
                    end
                end
            end
        end
    end
    return count / candidates
end

random_node_sample(graph) = sample(1:nv(graph), 2, replace=false)

function mapValues(x, y)
    if x == nothing || y == nothing
        return nothing
    end
    return  convert(Int64, (x >= y)) 
end

function unreliableClassification(graph, meanDegree)
    compSum, count = 0, 0
    numNodes = nv(graph)
    while count < 30
        u, v = random_node_sample(graph)
        x = vertexCompare(graph,u,v)
        compSum += (x == nothing) ? 0 : x
        count += (x == nothing) ? 0 : 1
    end

    compMean = compSum/count
    refFound = false
    
    u, v = nothing, nothing
    while !refFound
        u, v = random_node_sample(graph)
        num_friends_u = length(neighbors(graph, u))
        num_friends_v = length(neighbors(graph, v))
        if (num_friends_v <= meanDegree || num_friends_u <= meanDegree)
            continue
        end
        x = vertexCompare(graph, u, v)
        refFound = ((x != nothing) && x < compMean)
    end 

    output=[]
    
    for i=1:numNodes
        x, y= vertexCompare(graph, u, i), vertexCompare(graph, v, i)
        push!(output, mapValues(x, y))
    end

    return output
end

function evaluateAgreement(class1, class2, class3)
    return sum(1.0*(class1 .== class2 .== class3))
end

#Runs classify 3 times and refines these classification using the classifications of the vertices' neighbors.
#Then it assigns each vertex a community based on a majority vote.
function repeat_unreliable_class(graph, meanDegree, numRefine=)7
    classes = []
    for nT=1:3
        println("Trial $nT.....")
        println("Computing unreliable classification ....")
        curr_result = unreliableClassification(graph, meanDegree)
        println("....................................  Done")
        for _ in 1:numRefine # 7 refinement steps
            newClasses = -1*ones(length(curr_result))
            for i=1:nv(graph)
                diff = sum(2*filter(x->x!=nothing, curr_result[neighbors(graph,i)])-1)
                if diff > 0
                    newClasses[i] = 1
                elseif diff < 0
                    newClasses[i] = 0
                elseif diff == 0
                    newClasses[i] = sample(0:1)
                end
            end
            curr_result = newClasses
        end
        push!(classes, copy(curr_result))
    end

    c1, c2, c3 = classes
    # computing majority over permutations
    c2_swap = 1 - c2
    c3_swap = 1 - c3
    ag_max = -1
    ag1 = evaluateAgreement(c1, c2, c3)
    ag2 = evaluateAgreement(c1, c2, c3_swap)
    ag3 = evaluateAgreement(c1, c2_swap, c3)
    ag4 = evaluateAgreement(c1, c2_swap, c3_swap)
    if (ag1 > ag_max)
        ag_max = ag1 
        best = (c1, c2, c3)
    end
    if (ag2 > ag_max)
        ag_max = ag2 
        best = (c1, c2, c3_swap)
    end
    if (ag3 > ag_max)
        ag_max = ag3 
        best = (c1, c2_swap, c3)
    end
    if (ag4 > ag_max)
        ag_max = ag4 
        best = (c1, c2_swap, c3_swap)
    end
    return 1.0*(mean(best) .> 0.5)
end
    # c1, c2, c3 = classes

    # final1, final2, final3 = nothing, nothing, nothing 
    # for _ in range

    # # computing r12
    # matches, disagreements = 0, 0
    # for i=1:nv(graph)
    #     if c1[i] != nothing && c2[i] != 1 && c1[i] == c2[i]
    #         matches += 1
    #     end
    #     if c1[i] != nothing && c2[i] != 1 && (c1[i] + c2[i] == 1)
    #         disagreements += 1
    #     end
    # end
    # r12 = (matches > disagreements) ?  1 : -1

    # # computing r13
    # matches, disagreements = 0, 0
    # for i=1:nv(graph)
    #     if c1[i] != nothing && c3[i] != 1 && c1[i] == c3[i]
    #         matches += 1
    #     end
    #     if c1[i] != nothing && c3[i] != 1 && (c1[i] + c3[i] == 1)
    #         disagreements += 1
    #     end
    # end
    # r13 = (matches > disagreements) ?  1 : -1

    # # computing r23
    # matches, disagreements = 0, 0
    # for i=1:nv(graph)
    #     if c2[i] != nothing && c3[i] != 1 && c2[i] == c3[i]
    #         matches += 1
    #     end
    #     if c2[i] != nothing && c3[i] != 1 && (c2[i] + c3[i] == 1)
    #         disagreements += 1
    #     end
    # end
    # r23 = (matches > disagreements) ?  1 : -1

    # if (r12*r13*r23 == - 1)
    #     if (r23 < r13 && r12 < r23)
    #         r12 = -r12
    #     elseif (r13 < r23)
    #         r13 = -r13
    #     else
    #         r23 = -r23
    #     end
    # end

    # output = []
    # for i=1:nv(graph)
    #     if(c1[i] != nothing && c2[i] != nothing && c3[i] != nothing)
    #         balance = c1[i] + r12*c2[i]+r13*c3[i]
    #         class = (balance > 0) ?  1 : 0
    #         push!(output, class)
    #     else
    #         push!(output, nothing)
    #     end
    # end

#     return output
# end

