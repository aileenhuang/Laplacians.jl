<!-- AUTOGENERATED. See 'doc/build.jl' for source. -->
The following is a list of the graph generators.

## Deterministic
### completeGraph
```
completeGraph(n::Int64)
```
The complete graph 

### pathGraph
```
pathGraph(n::Int64)
```
The path graph on n vertices 

### ringGraph
```
ringGraph(n::Int64)
```
The simple ring on n vertices

### generalizedRing
```
generalizedRing(n::Int64, gens)
```
A generalization of a ring graph. The vertices are integers modulo n. Two are connected if their difference is in gens. For example, 

```
generalizedRing(17, [1 5])
```

### hyperCube
```
hyperCube(d::Int64)
```
The d dimensional hypercube.  Has 2^d vertices

### completeBinaryTree
```
completeBinaryTree(n::Int64)
```
The complete binary tree on n vertices

### grid2
```
grid2(n::Int64)
grid2(n::Int64, m::Int64)
```
An n-by-m grid graph.  iostropy is the weighting on edges in one direction.

### grid2coords
```
grid2coords(n::Int64, m::Int64)
grid2coords(n)
```
Coordinates for plotting the vertices of the n-by-m grid graph

## Random
These are randomized graph generators.
            ### randMatching
```
randMatching(n::Int64)
```
A random matching on n vertices

### randRegular
```
randRegular(n::Int64, k::Int64)
```
A sum of k random matchings on n vertices

### grownGraph
```
grownGraph(n::Int64, k::Int64)
```
Create a graph on n vertices. For each vertex, give it k edges to randomly chosen prior vertices. This is a variety of a preferential attachment graph.    

### grownGraphD
```
grownGraphD(n::Int64, k::Int64)
```
Like a grownGraph, but it forces the edges to all be distinct. It starts out with a k+1 clique on the first k vertices

### prefAttach
```
prefAttach(n::Int64, k::Int64, p::Float64)
```
A preferential attachment graph in which each vertex has k edges to those that come before.  These are chosen with probability p to be from a random vertex, and with probability 1-p to come from the endpoint of a random edge. It begins with a k-clique on the first k+1 vertices.
