# lexconvergence.jl
#
# Contains various convergence tests and plots on
# IterLex, proposed by Xiao Shi.
# Includes implementation of quadratic minimization.
#
# Created by Aileen Huang, Spring 2017
import Laplacians.intHeap
import Laplacians.intHeapAdd!, Laplacians.intHeapPop!
# using PyPlot
using Laplacians
using Plots
include("../src/lex.jl")

ITERS = 500

#=================================ALGORITHM SIMULATIONS===============================#

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
      # @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
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
      # @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
      return [val, n, t, EPSILON]
    end

    tmp = val
    val = nextVal
    nextVal = tmp
    t=t+1
  end
  return [val, n, t, EPSILON]
end

function simIterQuadUnwtd{Tv, Ti}(numIter::Int64,
                                 A::SparseMatrixCSC{Tv, Ti},
                                 isTerm::Array{Bool, 1},
                                 initVal::Array{Float64, 1}, )
  n = A.n
  val = copy(initVal)
  nextVal = zeros(Float64, n)

  for t = 1:numIter
    # if the bits representation of val and nextVal are the same for
    # ever vertex, then there is no point in keeping iterating.
    progress = false
    for u = 1:n         # For each node, perform the algorithm...
      if (!isTerm[u])
        nbrs = A.rowval[A.colptr[u]:(A.colptr[u + 1] - 1)]

        avgsum = float(0)
        for i in val[nbrs]
          avgsum += i
        end 
        nextVal[u] = avgsum / length(nbrs)
        if (bits(val[u]) != bits(nextVal[u]))
          progress = true
        end
      else
        nextVal[u] = val[u]
      end
    end

    if (!progress)
      @printf("INFO: simIterLexUnwtd: terminating early after %d iterations,
              as numerical error prevents further progress.\n", t)
      return val
    end

    tmp = val
    val = nextVal
    nextVal = tmp
  end
  return val
end


# ============================= TESTER FUNCTIONS ==================================
# Plot wrapper
# function plotWrapper(fxn, xlabel, ylabel)
#     plt = Plots.plot(title = title, xlabel=xlabel, ylabel=ylabel)
#     if fxn == "chimera"
#         dim_list, lex_iterations, quad_iterations = chimeraTester(10000, 10, 10)
#         plotRuns(dim_list, quad_iterations, lex_iterations, "Experiments on Randomly Generated Chimera Graphs", "Size of Graph, n", "Num Iters")
#     end
    
# end


# Tests on grid2 graph generation.
#
# n and m are the dims of the grid
# numterms = number of terminal nodes
# startn = initial value of n
# startm = intial value of m
function grid2Tester(n::Int64, m::Int64, numterms::Int64, startn::Int64)
    dim_list = []
    quad_iterations = []
    lex_iterations = []

    i = startn
    while i <= n
        j = i
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
            push!(lex_iterations, lexVolt[3])
            push!(quad_iterations, quadVolt[3])
            j = j*2
        end
        i = i*2
    end
    return [dim_list, lex_iterations, quad_iterations]

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
        push!(lex_iterations, lexVolt[3])
        push!(quad_iterations, quadVolt[3])

        i = i*2
    end
    return [dim_list, lex_iterations, quad_iterations]
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
        push!(lex_iterations, lexVolt[3])
        push!(quad_iterations, quadVolt[3])
        i = i*2
    end
    return [dim_list, lex_iterations, quad_iterations]

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
        graph = randGenRing(i, k)
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
        push!(lex_iterations, lexVolt[3])
        push!(quad_iterations, quadVolt[3])
        i = i*2
    end
    return [dim_list, lex_iterations, quad_iterations]
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
        graph = grownGraph(i, k)
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
        push!(lex_iterations, lexVolt[3])

        push!(quad_iterations, quadVolt[3])
        i = i*2
    end
    return [dim_list, lex_iterations, quad_iterations]

end

# Creates a sample (fairly hardcoded) grid2 graph for easy testing
function makeSampleGridGraph()
    #create a generic grid graph
    i = 100
    j = 100

    dim = i*j
    graph = grid2(i::Int64, j::Int64; isotropy=1)
    isTerm = zeros(Bool, dim)
    initVal = zeros(dim)
    numterms = 10
    numIter = ITERS

    perm = randperm(i*j)[1:numterms]
    for elt in perm
        isTerm[elt] = true
        initVal[elt] = rand(1)[1]
    end
    return [dim, graph, isTerm, initVal]
end

# Creates a sample (fairly hardcoded) randRegular graph for easy testing
function makeSampleRandRegular()
    dim = 10000
    k = 100
    graph = randRegular(dim, k)
    isTerm = zeros(Bool, dim)
    initVal = zeros(dim)
    numterms = 10

    perm = randperm(dim)[1:numterms]
    for elt in perm
        isTerm[elt] = true
        initVal[elt] = rand(1)[1]
    end
    return [dim, graph, isTerm, initVal]
end

#=============================== TESTS AGAINST LEX.JL ============================#

# Looks at the maximum difference between potentials and true solution at every iteration
# For IterLex
function maxLexDifference(dim, graph, isTerm, initVal)
    #create a generic grid graph
    numIter = ITERS
    
    sol = simIterLexUnwtd(ITERS, graph, isTerm, initVal)
    
    val = copy(initVal)
    nextVal = zeros(Float64, dim)
    EPSILON = 1/convert(Float64, dim)
    t = 1
    difference = []
    while t <= numIter
#         # If all nodes change within some epsilon
#         # then there is no point in continuing.
        append!(difference, maximum(sol - val))
        progress = false
            for u = 1:dim
              if (!isTerm[u])
                nbrs = graph.rowval[graph.colptr[u]:(graph.colptr[u + 1] - 1)]
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
        # @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
            # plot(difference, label="Difference between solution and potentials per iteration")
          return difference
        end

        tmp = val
        val = nextVal
        nextVal = tmp
        t = t+1
    end
    # plot(difference, label="Difference between solution and potentials per iteration")
    return difference
end

# Looks at the maximum difference between potentials and true solution at every iteration
# For IterQuad
function maxQuadDifference(dim, graph, isTerm, initVal)
    #create a generic grid graph
    numIter = ITERS
    sol = simIterQuadUnwtd(ITERS, graph, isTerm, initVal)
    
    val = copy(initVal)
    nextVal = zeros(Float64, dim)
    EPSILON = 1/convert(Float64, dim)
    t = 1
    difference = []
    while t <= numIter
        # If all nodes change within some epsilon
        # then there is no point in continuing.
        append!(difference, maximum(sol - val))
        progress = false
            for u = 1:dim         # For each node, perform the algorithm...
                if (!isTerm[u])
                    nbrs = graph.rowval[graph.colptr[u]:(graph.colptr[u + 1] - 1)]
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

        if (!progress)
          @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
            # plot(difference, label="Difference between solution and potentials per iteration")
          return difference
        end

        tmp = val
        val = nextVal
        nextVal = tmp
        t = t+1
    end
#     plot(difference, label="Difference between solution and potentials per iteration")
    return difference
end

# Looks at the maximum difference between iterations
function maxGrowthPerIteration()
    #create a generic grid graph
    i = 100
    j = 100

    dim = i*j
    graph = grid2(i::Int64, j::Int64; isotropy=1)
    # graph = chimera(dim)
    isTerm = zeros(Bool, dim)
    initVal = zeros(dim)
    numterms = 10
    numIter = ITERS
    
    perm = randperm(i*j)[1:numterms]
    for elt in perm
        isTerm[elt] = true
        initVal[elt] = rand(1)[1]
    end
    
    val = copy(initVal)
    nextVal = zeros(Float64, dim)
    EPSILON = 1/convert(Float64, dim)
    t = 1
    growth = []
    while t <= numIter
#         # If all nodes change within some epsilon
#         # then there is no point in continuing.

        progress = false
            for u = 1:dim
              if (!isTerm[u])
                nbrs = graph.rowval[graph.colptr[u]:(graph.colptr[u + 1] - 1)]
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
        append!(growth, maximum(nextVal - val))

        if (!progress)
          # @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
          return growth
        end

        tmp = val
        val = nextVal
        nextVal = tmp
        t = t+1
    end
    return growth
end

#=========================================PLOTTING CODE ==================================#
function plotRuns(dim_list, lex_iterations, quad_iterations)
    Plots.plot!(dim_list, lex_iterations, linecolor="orange", label="lex")
    Plots.plot!(dim_list, quad_iterations, linecolor="blue", label="quadratic")
end

# Plots average runtimes vs. graph size for IterQuad and IterLex
# by running n_experiments trials on tester functions
function plotAverageRuns(num_nodes, numterms, startn, fxn, n_experiments, xlabel, ylabel)
    if fxn != "grid2"
        num_data_pts = Int(ceil(log2(num_nodes/startn)))
    else
        num_data_pts = Int(ceil(log2((num_nodes^2)/startn)))
    end

    sum_lex = zeros(num_data_pts)
    sum_quad = zeros(num_data_pts)
    
    lex_trials = []
    quad_trials = []
    wrapper = []
    
    i = 1
    while i <= num_data_pts
       push!(lex_trials, [])
       push!(quad_trials, [])
        i += 1
    end

    i = 1
    while i <= n_experiments
        #switch case for functions
        if fxn == "chimera"
            wrapper = chimeraTester(num_nodes, numterms, startn)
        elseif fxn == "grid2"
            wrapper = grid2Tester(num_nodes, num_nodes, numterms, startn)
        elseif fxn == "randRegular"
            wrapper = randRegularTester(num_nodes, Int(num_nodes/100), numterms, startn)
        elseif fxn == "randGenRing"
            wrapper = randGenRingTester(num_nodes, Int(num_nodes/100), numterms, startn)
        elseif fxn == "grownGraph"
            wrapper = grownGraphTester(num_nodes, Int(num_nodes/100), numterms, startn)
        end
        
        dim_list = wrapper[1]
        single_lex = wrapper[2]
        single_quad = wrapper[3]
            
        j = 1
        while j <= num_data_pts
            push!(lex_trials[j], single_lex[j])
            push!(quad_trials[j], single_quad[j])
            j+=1
        end
        sum_lex += wrapper[2]
        sum_quad += wrapper[3]
        i += 1
    end
    
    if fxn == "chimera"
        title = string("Average Runs on Randomly Generated Chimera Graphs, ", string(n_experiments), " trials")
    elseif fxn == "grid2"
        title = string("Average Runs on Randomly Generated Grid2 Graphs, ", string(n_experiments), " trials")
    elseif fxn == "randRegular"
        title = string("Average Runs on Randomly Generated randRegular Graphs, ", string(n_experiments), " trials")
    elseif fxn == "randGenRing"
        title = string("Average Runs on Randomly Generated randGenRing Graphs, ", string(n_experiments), " trials")
    elseif fxn == "grownGraph"
        title = string("Average Runs on Randomly Generated grownGraph Graphs, ", string(n_experiments), " trials")
    end

    plt = Plots.plot(title = title, xlabel=xlabel, ylabel=ylabel)
    
    avg_lex = sum_lex/n_experiments
    avg_quad = sum_quad/n_experiments
    dim_list = wrapper[1]
    plotRuns(dim_list, avg_lex, avg_quad)
    display(plt)
    return [dim_list, num_data_pts, lex_trials, quad_trials, avg_lex, avg_quad]
end

# Plots the standard deviations of runtimes with respect to graph size
# for both IterLex and IterQuad
function plotStandardDevs(dim_list, num_data_pts, lex_trials, quad_trials, avg_lex, avg_quad)
    lex_stds = []
    quad_stds = []
    i=1
    while i <= num_data_pts
        lex_elt = lex_trials[i]
        quad_elt = quad_trials[i]

        push!(lex_stds, stdm(lex_elt, avg_lex[i]))
        push!(quad_stds, stdm(quad_elt, avg_quad[i]))

        i+=1
    end
    plt2 = Plots.plot(title = "Standard Deviation vs Graph Size", xlabel="n", ylabel="Standard Deviation")
    Plots.plot!(dim_list, lex_stds, linecolor="red", label="lex")
    Plots.plot!(dim_list, quad_stds, linecolor="green", label="quadratic")
    display(plt2)
end

# Plots the maximum difference per iteration for IterLex and QuadLex
# based on a predetermined graph
function plotMaxDifference(lex_difference, quad_difference)
    plt = Plots.plot(title="Difference, randRegular", xlabel="num iters", ylabel="max difference")
    Plots.plot!(lex_difference, label="lex max difference")
    Plots.plot!(quad_difference, label="quad max difference")
    display(plt)
end

#Plots the maximum growth for a grid2 graph per iteration for IterLex
function plotMaxGrowth()
    i = 1
    plt = Plots.plot(title="Max Growth Per Iteration", xlabel="Iteration", yscale = :log, ylabel="growth")

    while i <= 10
        growth = maxGrowthPerIteration()
        Plots.plot!(growth)
        i += 1
    end
    display(plt)
end

# # Looks at the maximum difference between potentials and true solution at every iteration
# # Need to clean up and figure what to do with this later
# function convexityAnalysis()
#     #create a generic grid graph
#     i = 100
#     j = 100

#     dim = i*j
# #     graph = grid2(i::Int64, j::Int64; isotropy=1)
#     graph = chimera(dim)
#     isTerm = zeros(Bool, dim)
#     initVal = zeros(dim)
#     numterms = 10
#     numIter = ITERS
    
#     perm = randperm(i*j)[1:numterms]
#     for elt in perm
#         isTerm[elt] = true
#         initVal[elt] = rand(1)[1]
#     end
    
#     val = copy(initVal)
#     nextVal = zeros(Float64, dim)
#     EPSILON = 1/convert(Float64, dim)
#     t = 1
# #     growth = []
#     while t <= numIter
# #         # If all nodes change within some epsilon
# #         # then there is no point in continuing.

#         progress = false
#             for u = 1:dim
#               if (!isTerm[u])
#                 nbrs = graph.rowval[graph.colptr[u]:(graph.colptr[u + 1] - 1)]
#                 maxNeighbor = maximum(val[nbrs])
#                 minNeighbor = minimum(val[nbrs])
#                 nextVal[u] = minNeighbor + (maxNeighbor - minNeighbor) / 2.0
#                 if (nextVal[u] - val[u] > EPSILON)
#                   progress = true
#                 end
#               else
#                 nextVal[u] = val[u]
#               end
#             end
#         plot(cumsum(sort(val)))

#         if (!progress)
#           @printf("Terminated after %d iterations after getting within epsilon = %f.\n", t, EPSILON)
#         end

#         tmp = val
#         val = nextVal
#         nextVal = tmp
#         t = t+1
#     end
# end

# convexityAnalysis()

# An attempt at a three-dimensional plotter to show
# the relationship between n and k for the Tester functions; not working yet, need to revisit
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