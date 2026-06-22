using Random
using LinearAlgebra

# Parameters
Nparticles = 100000          # number of particles
Rrelease   = 200            # release radius
Rkill      = 2Rrelease     # kill radius
# dirs = [(1,0), (-1,0), (0,1), (0,-1)]
dirs = [(0, 1), (-1,0), (1, 0)]
# Cluster stored as a Set of lattice points
cluster = Set{Tuple{Int,Int}}()
push!(cluster, (0,0))      # seed

# Helper functions
function neighbors(x, y)
    return ((x+dx, y+dy) for (dx,dy) in dirs)
end

function touches_cluster(x, y, cluster)
    for nb in neighbors(x,y)
        nb in cluster && return true
    end
    return false
end

for n in 1:Nparticles
    # launch point on circle
    θ = 2π * rand()
    x = round(Int, Rrelease * cos(θ))
    y = round(Int, Rrelease * sin(θ))

    while true
        # random walk step
        dx, dy = rand(dirs)
        x += dx
        y += dy

        r = sqrt(x^2 + y^2)

        # kill if too far
        if r > Rkill
            break
        end

        # stick if touching cluster
        if touches_cluster(x, y, cluster)
            push!(cluster, (x,y))
            break
        end
    end

    n % 5000 == 0 && println("Particles: $n")
end

using Plots

xs = [p[1] for p in cluster]
ys = [p[2] for p in cluster]

scatter(xs, ys;
    markersize=1,
    aspect_ratio=:equal,
    legend=false)
    # title="2D Diffusion-Limited Aggregation")
