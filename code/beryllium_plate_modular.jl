#=

# Bending of berrylium plate 

=#


module plate 
using SmoothedParticles
# import SmoothedParticles
using Parameters
using Plots
using CSV
using DataFrames
import LinearAlgebra
include("algebra.jl")

#
#CONSTANT PARAMETERS
#-------------------------------
#
abstract type DistortionModel end
abstract type DoubleSumModel <: DistortionModel end

struct SingleSum <: DistortionModel end
struct SingleSumEvolved <: DistortionModel end
struct DoubleSum <: DoubleSumModel end
struct DoubleSumEvolved <: DoubleSumModel end

const MODEL = SingleSumEvolved()
const folder_name = "final_report/"*string(nameof(typeof(MODEL)))

const L = 0.06   #rod length
const W = 0.01   #rod width

# const pull_time = 0.5  #for how long we pull

const c_l = 9046.59   #longitudinal sound speed
const c_s = 9046.59  #shear sound speed
const c_0 = sqrt(c_l^2 + 4/3*c_s^2)  #total sound speed 
const rho0 = 1845.0   #density
const nu = 1.0e-4    #artificial viscosity (surpresses noise but is not neccessary)
const c_p = 0.10*4.0*c_l  #tensile penalty term
# const c_p = 0.075*4.0*c_l  #tensile penalty term
const c_shift = 0.000005*0.5           #particle shifting factor

const dr = W/40    #discretization step
# const dr = W/80    #discretization step
# const h = (3.5+10*eps(Float64))dr    #support radius
const h = (3.0+10*eps(Float64))dr    #support radius
const h2 = h*h
const vol = dr^2   #particle volume
const m = rho0*vol #particle mass
# const D = 0.05*0.5*h^2  #diffusion coefficient for shifting
# const eta = h/dr

const dt = 0.01dr/c_0 #time step 
const t_end = 3e-5/1.0  #total simulation time
const dt_plot = max(t_end/400, dt) #how often save txt data (cheap)
const dt_frame = max(t_end/100, dt) #how often save pvd data (expensive)

#ALGEBRAIC TOOLS
#----------------------------------------

@inbounds function outer(x::RealVector, y::RealVector)::FlatMatrix
    return FlatMatrix(
        x[1]*y[1], x[2]*y[1],  
        x[1]*y[2], x[2]*y[2], 
    )
end

#
#DEFINE VARIABLES
#------------------------------

@with_kw mutable struct Particle <: AbstractParticle
	x::RealVector         # position
    x_avg::RealVector = VEC0
    v::RealVector = init_velocity(x) # velocity
    f::RealVector = VEC0  # force
    X::RealVector = x     # Lag. position
    X_avg::RealVector = VEC0     # mean Lag. position
    A::FlatMatrix = FMAT1  # distortion
    P::FlatMatrix = FMAT0
    Q::FlatMatrix = FMAT0
    invQ::FlatMatrix = FMAT0
    pressure::Float64 = 0.0 
    dev_stress::FlatMatrix = FMAT0
    rho::Float64 = 0.0    # density
    C_rho::Float64 = 0.0    # inital offset of density 
    lambda::Float64 = 0.0
    C_lambda::Float64 = 0.0 # initial offset of lambda
    ker_sum::Float64 = 0.0
    L::FlatMatrix = FMAT1 #velocity gradient
    # Diagnostics
    e::Float64 = 0.       # fronorm squared of eta
    error::RealVector = VEC0
    condP::Float64 = 0.0    # the condition number of P
    condQ::Float64 = 0.0    # the condition number of Q
    eQ::Float64 = 0.0
    # Energies
    E_kinet::Float64 = 0.0
    E_shear::Float64 = 0.0
    E_vol::Float64 = 0.0
    E_penalty::Float64 = 0.0
    G::FlatMatrix = FMAT0
    norm_devG::Float64 = 0.0
end

include("distortion_SingleSumEvolved.jl")
include("distortion_SingleSum.jl")
include("distortion_DoubleSumEvolved.jl")
include("distortion_DoubleSum.jl")

#STRUCTURAL KERNELS
#------------------

@fastmath function wendland2h(h::Float64, r::Float64)::Float64
    x = r/h
    return x < 1.0 ? 14.0*(1.0 - x)^3*(14.0*x^2 - 3.0*x - 1.0)/(pi*h^2) : 0.
end

@fastmath function rDwendland2h(h::Float64, r::Float64)::Float64
    x = r/h
    return x < 1.0 ? 140.0*(1.0 - x)^2*(4.0 - 7.0*x)/(pi*h^4) : 0.
end

#CREATE INITIAL STATE
#----------------------------

function make_geometry()
    grid = Grid(dr, :hexagonal)
    # grid = Grid(dr, :square)
    rod = Rectangle(-L/2, -W/2, L/2, W/2)
    dom = BoundaryLayer(rod, grid, L + W)
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, rod, x -> Particle(x=x))
    create_cell_list!(sys)

    apply!(sys, find_rho!)
    for p in sys.particles 
        p.C_rho = rho0 - p.rho
        p.rho = 0.0
        p.C_lambda = 0.0 - p.lambda
        # p.C_lambda = 1.0 - p.ker_sum
        p.ker_sum = 0.0
    end
    apply!(sys, find_rho!)
    force_computation!(MODEL, sys)

    return sys
end

function init_velocity(x::RealVector)::RealVector
    A = 4.3369e-5
    omega = 2.3597e5
    alpha = 78.834
    a1 = 56.6368
    a2 = 57.6455
    s = alpha*(x[1] + L/2)
    v = A*omega*(a1*(sinh(s) + sin(s)) - a2*(cosh(s) + cos(s)))
    return 1*v*VECY
end

#
#PHYSICS
#-------------------------------------

function find_rho!(p::Particle, q::Particle, r::Float64) 
    p.rho += m*wendland2(h,r)
    p.lambda+= m*wendland2h(h,r)

    ker = wendland2(h,r)
    # find the avg x_avg of x * ker_sum
    p.x_avg += ker*(q.x - p.x)   # accumulate relative positions
    # p.x_avg += ker*q.x
    p.X_avg += ker*(q.X - p.X)
    # p.X_avg += ker*q.X
    # find the kernel sum
    p.ker_sum += ker

end 

function find_avg!(p::Particle)
    # p.ker_sum += p.C_lambda
    # p.x_avg += wendland2(h, 0.0)*p.x  # self contribution
    # p.X_avg += wendland2(h, 0.0)*p.X
    # get x_avg and X_avg
    p.x_avg = p.x_avg / p.ker_sum
    p.X_avg = p.X_avg / p.ker_sum
end

function update_v!(p::Particle)
    p.v += 0.5*dt*p.f/m
end

function update_x!(p::Particle)
    p.x += 0.5*dt*p.v
end

function reset!(p::Particle)
    # reset ALL per-step accumulators 
    p.f    = VEC0
    p.e    = 0.0
    p.P    = FMAT0
    p.Q    = FMAT0
    p.L = FMAT0
    p.eQ   = 0.0
    p.rho  = 0.0 
    p.lambda = 0.0

    # reset the averaging values
    p.ker_sum = 0.0
    p.x_avg = VEC0
    p.X_avg = VEC0
end

#
# Diagnostics
# ------------------------------------------------
function particle_energy(p::Particle, N::Int, E0::Float64)
    G0 = dev(p.G)
    E_kinet = 0.5*m*dot(p.v, p.v)
    # E_shear = 0.25*m*c_s^2*LinearAlgebra.norm(G0,2)^2
    E_shear = 0.25*m*c_s^2*norm(G0)^2
    E_vol = 0.5*m*c_0^2*(rho0/p.rho - 1)^2
    E_penalty = 0.5*m*c_p^2*(p.lambda/rho0)^2
    p.E_kinet = E_kinet*N/E0
    p.E_shear = E_shear*N/E0
    p.E_vol = E_vol*N/E0
    p.E_penalty = E_penalty*N/E0
    return E_kinet + E_shear + E_vol + E_penalty
end

#
#TIME ITERATION
#--------------

function integration_step(sys::ParticleSystem)
    #verlet scheme
    apply!(sys, update_v!)
    apply!(sys, update_x!)
    evolve_A!(MODEL, sys)
    apply!(sys, update_x!)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, find_rho!)
    force_computation!(MODEL, sys)
    apply!(sys, update_v!)
end

function main()
    println("Simulation starting")
    sys = make_geometry()
    out = new_pvd_file("results/"*folder_name)
    csv_data = open("results/"*folder_name*"/plate.csv", "w")

    p_sel = argmax(p -> abs(p.x[1]) + abs(p.x[2]), sys.particles) 

    E0 = sum(p -> particle_energy(p, length(sys.particles), 1.0), sys.particles) 
    println("Initial energy E0 = ", E0)
    # E0 = 1/length(sys.particles)

    # !!! Warning !!!
    # For stationary cases norming with energy E0 is not adviced as E0 is almost zero!
    E0 = 1.0

    
    @time for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        # if(k == 100)
        #     E0 = sum(p -> particle_energy(p, length(sys.particles), E0), sys.particles) 
        #     println("Initial energy E0 = ", E0)
        # end

        if (k % Int64(round(dt_plot/dt)) == 0)
            println("t = ", k*dt)
            println("N = ", length(sys.particles))
            E = sum(p -> particle_energy(p, length(sys.particles), E0), sys.particles)
            println("E = ", E/E0)
            # println("
            # println("h = ", p_sel.x[2])
            println()
            write(csv_data, string(k*dt,",",p_sel.x[2],",",E/E0,"\n"))
            # write(csv_data, string(k*dt,",",p_sel.x[2],",",E/E0-1,"\n"))
            # numerical_force(sys)
        end
        if (k % Int64(round(dt_frame/dt)) == 0)
            # apply!(sys, find_e!)
            save_frame!(out, sys, :v, :A, :e, :P, :Q, :invQ, :L,
                        :eQ, :error, :x_avg,
                        :pressure, :dev_stress, :norm_devG, :G,
                        :rho, :C_rho, :lambda, 
                        :C_lambda, :ker_sum,
                        :E_kinet, :E_shear, :E_vol, :E_penalty, 
                        :f, :condP, :condQ)#, :dv, :h, :Omega)
        end
        integration_step(sys)
    end
    save_pvd_file(out)
    close(csv_data)
    data = CSV.read("results/"*folder_name*"/plate.csv", DataFrame; header=false)
    p1 = plot(data[:,1], data[:,2], label = "plate-test", xlabel = "time", ylabel = "amplitude")
    p2 = plot(data[10:end,1], data[10:end,3], label = "energy",
              xlabel = "time", ylabel = "energy", ylims=(0,1.2))
    savefig(p1, "results/"*folder_name*"/amplitude.pdf")
    savefig(p2, "results/"*folder_name*"/energy.pdf")
end

# end # module end



# Extension of apply operator for ternary functions
#--------------------------------------------------
import SmoothedParticles: AbstractParticle, ParticleSystem

function dist2(p::AbstractParticle, q::AbstractParticle)::Float64
    return dot(p.x - q.x, p.x - q.x)
end 

@inline function _apply_ternary!(sys::ParticleSystem, action!::Function, p::AbstractParticle)
    key = SmoothedParticles.find_key(sys, p.x)
    # collect neighbours of p
    # TODO: If the particles are non-penetrating, the number of neighbour particles should be fixed from 
    # above and we could preallocate the neighbour array for better performance.
    neighbours = Int[]
    for Δkey in sys.key_diff
        neigh_key = key + Δkey
        if 1 <= neigh_key <= sys.key_max
            for j in sys.cell_list[neigh_key].entries
                if j == 0; break; end
                q = sys.particles[j]
                if q == p; continue; end
                #rq = dist(p, q) 
                rq2 = dist2(p,q)
                #if rq <= sys.h
                if rq2 <= h2
                    push!(neighbours, j)
                end
            end
        end
    end
    # For each neighbour q, loop over all neighbours r 
    for k in neighbours 
        #if k == 0; break; end 
        q = sys.particles[k] 
        rq = dist(p, q) 
        for l in neighbours 
            #if l == 0; break; end 
            r = sys.particles[l]
            if r == p || r == q; continue; end 
            rr = dist(p, r) 
            action!(p, q, r, rq, rr)
        end 
    end 
end
"""
    apply_ternary!(sys::ParticleSystem, action!::Function)

Apply a ternary operator `action!(p::T, q::T, r::T, 
                                  rq::Float64, rr::Float64)` 
between particle `p` and all pairs of its neighbours `q` and `r`
in `sys::ParticleSystem{T}`. Values `rq` and `rr` are the
distances between `p` and `q` and between `p` and `r`, respectively.
This excludes all particles `q` and `r` with distance greater than `sys.h`.
This has complexity O(N*k^2) where N is the number of particles and k the
average number of neighbours per particle and runs in parallel.

!!! warning "Warning"
    Modifying particles `q` or `r` within `action!` can lead to race condition.
    Selecting large `sys.h` leads to significant performance drop.
"""
function apply_ternary!(sys::ParticleSystem, action!::Function)
    Threads.@threads for p in sys.particles
        _apply_ternary!(sys, action!, p)
    end 
end

end # module end
