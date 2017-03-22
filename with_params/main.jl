Pkg.add("LightGraphs")
include("helpers.jl")
using LightGraphs


println("Loading Political Blog Dataset ....")
original = loadgraph("data/polblogs.gml", :gml)
println("Finding Largest Connected Component")
ccs = weakly_connected_components(original)
largest_cc = ccs[indmax(map(x->size(x)[1], ccs))]
println("Obtaining Subgraph for Largest Connected Component")
blog_dataset, _ = induced_subgraph(original, largest_cc)
blog_dataset = Graph(blog_dataset)

println("Obtaining a classification of nodes")
meanDegree = sum(adjacency_matrix(blog_dataset))/(nv(blog_dataset))
println("Mean degree is $meanDegree")
for _=1:5
    println(sum(repeat_unreliable_class(blog_dataset, meanDegree)))
end