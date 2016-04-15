#=
 This is code for generating random trees of various sorts.
 The hope is that one might have low stretch,
 although none is guaranteed to.

 randishPrim produces lower stretch trees than randishKruskall
=#

using DataStructures

immutable ValWtPair
    val::Int64
    wt::Float64
end

import Base.isless

isless(e1::ValWtPair, e2::ValWtPair) = e1.wt < e2.wt


# sample edges with probability proportional to their weight

function randishKruskal{Tv,Ti}(mat::SparseMatrixCSC{Tv,Ti})

    a = copy(mat)
    numnz = nnz(a)

    r = -log(rand(numnz))
    for i in 1:numnz
        a.nzval[i] = r[i] / a.nzval[i]
    end

    tr1 = kruskal(a)
    unweight!(tr1)

    tr = tr1 .* mat

    return tr
end


function randishKruskalV1{Tv,Ti}(mat::SparseMatrixCSC{Tv,Ti})
    n = size(mat)[1]
    (ai,aj,av) = findnz(triu(mat))
    m = length(av)
    
    s = Sampler(av)

    comps = IntDisjointSets(n)

    treeinds = zeros(Int64,n-1)
    numintree = 0
    for j in 1:m
        i = pop!(s)
        if !DataStructures.in_same_set(comps,ai[i],aj[i])
            numintree = numintree+1
            treeinds[numintree] = i
            DataStructures.union!(comps,ai[i],aj[i])
        end
    end

    tree = sparse(ai[treeinds],aj[treeinds],av[treeinds],n,n)
    tree = tree + tree'

    return tree

end

function randishPrim{Tval,Tind}(mat::SparseMatrixCSC{Tval,Tind})

    n = mat.n
    m = nnz(mat)

    (ai, aj, av) = findnz(mat)

    flipInd = flipIndex(mat)
    
    visited = zeros(Bool,n)

    s = Vector{ValWtPair}()

    treeEdges = zeros(Tind,n-1)
    numEdges = 0    

    # should randomize the initial choice
    v = one(Tind)

    # always add edge by highest vertex number first
    for ind in mat.colptr[v]:(mat.colptr[v+1]-1)
        wt = mat.nzval[ind]
        Base.Collections.heappush!(s, ValWtPair(ind, -log(rand())/wt))
    end

    visited[v] = true

    while length(s) > 0

      valwt = Base.Collections.heappop!(s) #hog 
      edge = valwt.val

        v = aj[edge] #hog
        if visited[v]
            v = ai[edge]
        end

        if !visited[v]

            numEdges += 1
            treeEdges[numEdges] = edge

            
            for ind in mat.colptr[v]:(mat.colptr[v+1]-1)
                u = mat.rowval[ind]

                wt = mat.nzval[ind]
                useind = v < u ? ind : flipInd[ind]
                if !visited[u]
                    Base.Collections.heappush!(s, ValWtPair(useind, valwt.wt -log(rand())/wt))

                end
            end

            visited[v] = true
        end
1
    end

    treeEdges = treeEdges[1:numEdges]

    tr = submatrixCSC(mat,treeEdges)
    tr = tr + tr';

    return tr

end # randishPrim



function randishPrimV1{Tval,Tind}(mat::SparseMatrixCSC{Tval,Tind})

  n = mat.n
  m = nnz(mat)

  (ai, aj, av) = findnz(mat)
  flipInd = flipIndex(mat)

    
  visited = zeros(Bool,n)

  s = Sampler(m)

  treeEdges = zeros(Tind,n-1)
  numEdges = 0    

  # should randomize the initial choice
  v = one(Tind)

  # always add edge by highest vertex number first
  for ind in mat.colptr[v]:(mat.colptr[v+1]-1)
      wt = mat.nzval[ind]
      push!(s, ind, wt)
  end

  visited[v] = true

  while s.nitems > 0

    edge = pop!(s)

    v = aj[edge]
    if visited[v]
        v = ai[edge]
    end

    if !visited[v]

        numEdges += 1
        treeEdges[numEdges] = edge

        
        for ind in mat.colptr[v]:(mat.colptr[v+1]-1)
            u = mat.rowval[ind]

            wt = mat.nzval[ind]
            useind = v < u ? ind : flipInd[ind]
            if !visited[u]
                push!(s, useind, wt)

            end
        end

        visited[v] = true
    end

  end

  treeEdges = treeEdges[1:numEdges]

  tr = sparse(ai[treeEdges],aj[treeEdges],av[treeEdges],n,n)
  tr = tr + tr';

  return tr

end # randishPrim

