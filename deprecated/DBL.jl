module DBM

using Random, LinearAlgebra, StatsBase

# Grid size
const Nx, Ny = 301, 201

# DBM parameter (η ≈ 3 gives lightning-like paths)
const η = 3.2

# Potential field
ϕ = zeros(Float64, Nx, Ny)

# Occupancy grid (true = discharge channel)
channel = falses(Nx, Ny)

# Boundary conditions
ϕ[:, 1] .= 0.0              # ground
ϕ[:, end] .= 1.0            # top electrode

# Initial seed
seed_x = Nx ÷ 2
seed_y = Ny
# channel[seed_x, seed_y] = true
channel[2*(Nx ÷ 3), Ny] = true
channel[1*(Nx ÷ 3), Ny] = true

# Nearest-neighbor Laplace solver (Gauss–Seidel)
function solve_laplace!(ϕ, channel; iters=500)
    for _ in 1:iters
        for i in 2:Nx-1, j in 2:Ny-1
            if !channel[i, j]
                ϕ[i, j] = 0.25 * (
                    ϕ[i+1, j] + ϕ[i-1, j] +
                    ϕ[i, j+1] + ϕ[i, j-1]
                )
            end
        end
    end
end

# Compute candidate growth sites
function boundary_sites(channel)
    sites = []
    for i in 2:Nx-1, j in 2:Ny-1
        if !channel[i, j]
            if channel[i+1,j] || channel[i-1,j] ||
               channel[i,j+1] || channel[i,j-1]
                push!(sites, (i,j))
            end
        end
    end
    return sites
end

# Electric field magnitude (finite difference)
function Efield(ϕ, i, j)
    Ex = (ϕ[i+1,j] - ϕ[i-1,j]) / 2
    Ey = (ϕ[i,j+1] - ϕ[i,j-1]) / 2
    return sqrt(Ex^2 + Ey^2)
end

# Growth loop
steps = 800

for step in 1:steps
    # @show(step)
    solve_laplace!(ϕ, channel)
    ϕ[channel] .= 1.0

    sites = boundary_sites(channel)
    isempty(sites) && break

    weights = [Efield(ϕ, i, j)^η for (i,j) in sites]
    weights ./= sum(weights)

    idx = sample(1:length(sites), Weights(weights))
    i, j = sites[idx]

    channel[i, j] = true
end

end #module

using Plots
channel = DBM.channel
# heatmap(channel', aspect_ratio=1, c=:black, axis=false)
# heatmap(Float64.(channel)', aspect_ratio=1, c=:gray, axis=false)
heatmap(channel',
    c = :grays,
    clims = (0, 1),
    aspect_ratio = 1,
    axis = false)

