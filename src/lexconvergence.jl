# lexconvergence.jl
#
# Contains various convergence tests and plots on
# IterLex, proposed by Xiao Shi.
# Includes implementation of quadratic minimization.
#
# Started by Aileen Huang, Spring 2017

import Laplacians.intHeap
import Laplacians.intHeapAdd!, Laplacians.intHeapPop!
include("../src/lex.jl")

ITERS = 500


# numIter: Number of iterations
# A: SparseMatrix, the adjacency matrix representation of the graph
# isTerm: An array of bools that designate whether a node
# is a terminal node or not
# initVal: An array of Float64 that designate the potentials
# each node in the graph
#
# simIterLexUnwtdEps runs unweighted IterLex on a graph
# and terminates if no node changes more than EPSILON = 1/n,
# n being the number of nodes in the graph.
# 
# Returns n = number of nodes, t = number of iters, and eps
function simIterLexUnwtdEps{Tv, Ti}(numIter::Int64,
                                 A::SparseMatrixCSC{Tv, Ti},
                                 isTerm::Array{Bool, 1},
                                 initVal::Array{Float64, 1}, )
  n = A.n
  val = copy(initVal)
  nextVal = zeros(Float64, n)
  EPSILON = 1/convert(Float64, n)
    t = 1
    while t <= numIter
    # If all nodes change within some epsilon
    # then there is no point in continuing.
    progress = false
    for u = 1:n
      if (!isTerm[u])
        nbrs = A.rowval[A.colptr[u]:(A.colptr[u + 1] - 1)]
        maxNeighbor = maximum(val[nbrs])
        minNeighbor = minimum(val[nbrs])
        nextVal[u] = minNeighbor + (maxNeighbor - minNeighbor) / 2.0
        if (nextVal[u] - val[u] > EPSILON)
          progress = true
        end
      else
        nextVal[u] = val[u]
      end
    end

    if (!progress)
      @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
      return [val, n, t, EPSILON]
    end

    tmp = val
    val = nextVal
    nextVal = tmp
    t = t+1
  end
  return [val, n, t, EPSILON]
end

# numIter: Number of iterations
# A: SparseMatrix, the adjacency matrix representation of the graph
# isTerm: An array of bools that designate whether a node
# is a terminal node or not
# initVal: An array of Float64 that designate the potentials
# each node in the graph
#
# simIterQuadUnwtdEps runs unweighted IterQuad on a graph
# and terminates if no node changes more than EPSILON = 1/n,
# n being the number of nodes in the graph.
#
# 
# Returns n = number of nodes, t = number of iters, and eps
function simIterQuadUnwtdEps{Tv, Ti}(numIter::Int64,
                                 A::SparseMatrixCSC{Tv, Ti},
                                 isTerm::Array{Bool, 1},
                                 initVal::Array{Float64, 1}, )
  n = A.n
  val = copy(initVal)               #current values of graph
  nextVal = zeros(Float64, n)       #values of next iteration
  EPSILON = 1/convert(Float64, n)   #epsilon = 1/n

    t = 1
    while t <= numIter
    # If all nodes change within some epsilon
    # then there is no point in continuing.
    progress = false    # tracks whether algorithm should terminate
    for u = 1:n         # For each node, perform the algorithm...
      if (!isTerm[u])
        nbrs = A.rowval[A.colptr[u]:(A.colptr[u + 1] - 1)]

        avgsum = float(0)
        for i in val[nbrs]
          avgsum += i
        end 
        nextVal[u] = avgsum / length(nbrs)
        if (nextVal[u] - val[u] > EPSILON)
          progress = true
        end
      else
        nextVal[u] = val[u]
      end
    end

    if (!progress)      # Termination criterion
      @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
      return [val, n, t, EPSILON]
    end

    tmp = val
    val = nextVal
    nextVal = tmp
    t=t+1
  end
  return [val, n, t, EPSILON]
end

# ============================= TESTER FUNCTIONS ==================================
# Tests on grid2 graph generation.
#
# n and m are the dims of the grid
# numterms = number of terminal nodes
# startn = initial value of n
# startm = intial value of m
function grid2Tester(n::Int64, m::Int64, numterms::Int64, startn::Int64, startm::Int64)
    dim_list = []
    quad_iterations = []
    lex_iterations = []
    
    i = startn
    j = startm
    while i <= n
        while j <= m
            dim = i*j
            graph = grid2(i::Int64, j::Int64; isotropy=1)
            isTerm = zeros(Bool, dim)
            initVal = zeros(dim)
            perm = randperm(i*j)[1:numterms]
            for elt in perm
                isTerm[elt] = true
                initVal[elt] = rand(1)[1]
            end
            
            lexVolt = simIterLexUnwtdEps(ITERS, graph, isTerm, initVal)
            quadVolt = simIterQuadUnwtdEps(ITERS, graph, isTerm, initVal)
            push!(dim_list, i*j)
            push!(quad_iterations, quadVolt[3])
            push!(lex_iterations, lexVolt[3])
            j = j*2
        end
        i = i*2
    end
    loglog(dim_list, quad_iterations, label="quad")
    loglog(dim_list, lex_iterations, label="lex")
    legend()
end

# Tests on chimera graph generation.
#
# n is the number of nodes in the graph
# numterms = number of terminal nodes
# startn = initial value of n
function chimeraTester(n::Int64, numterms::Int64, startn::Int64)
    dim_list = []
    quad_iterations = []
    lex_iterations = []
    
    i = startn
    while i <= n
        graph = chimera(i)
        isTerm = zeros(Bool, i)
        initVal = zeros(i)
        perm = randperm(i)[1:numterms]
        for elt in perm
            isTerm[elt] = true
            initVal[elt] = rand(1)[1]
        end

        lexVolt = simIterLexUnwtdEps(ITERS, graph, isTerm, initVal)
        quadVolt = simIterQuadUnwtdEps(ITERS, graph, isTerm, initVal)
        push!(dim_list, i)
        push!(quad_iterations, quadVolt[3])
        push!(lex_iterations, lexVolt[3])
        i = i*2
    end
    loglog(dim_list, quad_iterations, label="quad")
    loglog(dim_list, lex_iterations, label="lex")
    legend()
end

# Tests on randRegular graph generation.
#
# n is the number of nodes in the graph
# numterms = number of terminal nodes
# startn = initial value of n
function randRegularTester(n::Int64, k::Int64, numterms::Int64, startn::Int64)
    dim_list = []
    quad_iterations = []
    lex_iterations = []
    
    i = startn
    while i <= n
        dim = i
        graph = randRegular(i, k)
        isTerm = zeros(Bool, dim)
        initVal = zeros(dim)

        perm = randperm(i)[1:numterms]
        for elt in perm
            isTerm[elt] = true
            initVal[elt] = rand(1)[1]
        end

        lexVolt = simIterLexUnwtdEps(ITERS, graph, isTerm, initVal)
        quadVolt = simIterQuadUnwtdEps(ITERS, graph, isTerm, initVal)
        push!(dim_list, i)
        push!(quad_iterations, quadVolt[3])
        push!(lex_iterations, lexVolt[3])
        i = i*2
    end
    loglog(dim_list, quad_iterations, label="quad")
    loglog(dim_list, lex_iterations, label="lex")
    legend()
end

# Tests on randGenRing graph generation.
#
# n is the number of nodes in the graph
# numterms = number of terminal nodes
# startn = initial value of n
function randGenRingTester(n::Int64, k::Int64, numterms::Int64, startn::Int64)
    dim_list = []
    quad_iterations = []
    lex_iterations = []
    
    i = startn
    while i <= n
        dim = i
        graph = randRegular(i, k)
        isTerm = zeros(Bool, dim)
        initVal = zeros(dim)

        perm = randperm(i)[1:numterms]
        for elt in perm
            isTerm[elt] = true
            initVal[elt] = rand(1)[1]
        end

        lexVolt = simIterLexUnwtdEps(ITERS, graph, isTerm, initVal)
        quadVolt = simIterQuadUnwtdEps(ITERS, graph, isTerm, initVal)
        push!(dim_list, i)
        push!(quad_iterations, quadVolt[3])
        push!(lex_iterations, lexVolt[3])
        i = i*2
    end
    loglog(dim_list, quad_iterations, label="quad")
    loglog(dim_list, lex_iterations, label="lex")
    legend()
end

# Tests on grownGraph graph generation.
#
# n is the number of nodes in the graph
# numterms = number of terminal nodes
# startn = initial value of n
function grownGraphTester(n::Int64, k::Int64, numterms::Int64, startn::Int64)
    dim_list = []
    quad_iterations = []
    lex_iterations = []
    
    i = startn
    while i <= n
        dim = i
        graph = randRegular(i, k)
        isTerm = zeros(Bool, dim)
        initVal = zeros(dim)

        perm = randperm(i)[1:numterms]
        for elt in perm
            isTerm[elt] = true
            initVal[elt] = rand(1)[1]
        end

        lexVolt = simIterLexUnwtdEps(ITERS, graph, isTerm, initVal)
        quadVolt = simIterQuadUnwtdEps(ITERS, graph, isTerm, initVal)
        push!(dim_list, i)
        push!(quad_iterations, quadVolt[3])
        push!(lex_iterations, lexVolt[3])
        i = i*2
    end
    return [dim_list, quad_iterations, lex_iterations]
#     loglog(dim_list, quad_iterations, label="quad")
#     loglog(dim_list, lex_iterations, label="lex")
    # legend()
end


# An attempt at a three-dimensional plotter to show
# the relationship between n and k; not working yet, need to revisit
# function ThreeDimTestPlotter(n::Int64, k::Int64, numterms::Int64, startn::Int64)
#     wrapper = []
#     quadTests = []
#     maxVal = 0
#     j = 1
#     while j <= k
# #         push!(wrapper, grownGraphTester(n, k, numterms, startn))
#         append!(quadTests, grownGraphTester(n, k, numterms, startn)[3])
#         print(quadTests)
#         j = j+1
#     end
    
#     rows = k
#     cols = Int(ceil(log2(n/startn)))
#     x = linspace(0, cols)
#     y = linspace(0, rows)
#     xgrid = repmat(x',n,1)
#     ygrid = repmat(y,1,n)
    
#     z = reshape(quadTests, rows, cols)
    
#     fig = figure("pyplot_surfaceplot",figsize=(10,10))
#     ax = fig[:add_subplot](2,1,1, projection = "3d")
#     ax[:plot_surface](xgrid, ygrid, z, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
# #     xlabel("X")
# #     ylabel("Y")
# #     title("Surface Plot")

# end
# ThreeDimTestPlotter(100, 3, 10, 10)

#=============================== TESTS AGAINST LEX.JL ============================

