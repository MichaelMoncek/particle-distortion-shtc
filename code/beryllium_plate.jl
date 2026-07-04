#=

# Bending of berrylium plate 

=#


module plate 
using SmoothedParticles
using Parameters
using Plots
using CSV
using DataFrames
import LinearAlgebra
include("algebra.jl")


#
# DISTORTION MODELS
#------------------------------------------------
abstract type DistortionModel end
abstract type DoubleSumModel <: DistortionModel end

struct SingleSum <: DistortionModel end
struct SingleSumEvolved <: DistortionModel end
struct DoubleSum <: DoubleSumModel end
struct DoubleSumEvolved <: DoubleSumModel end
struct MLSEvolved <: DistortionModel end

const MODEL = SingleSumEvolved()
const BASE_FOLDER = "final_report/"


#
#CONSTANT PARAMETERS
#-------------------------------
const L = 0.06   #rod length
const W = 0.01   #rod width

const c_l = 9046.59   #longitudinal sound speed
const c_s = 9046.59  #shear sound speed
const c_0 = sqrt(c_l^2 + 4/3*c_s^2)  #total sound speed 
const rho0 = 1845.0   #density
const nu = 1.0e-4    #artificial viscosity (surpresses noise but is not neccessary)
const c_p = 1.0*0.10*4.0*c_l  #tensile penalty term
# const c_p = 0.010*4.0*c_l  #tensile penalty term
const init_velocity_multiplier = 1.0

const dr = W/40    #discretization step
const h = (3.0+10*eps(Float64))dr    #support radius
const h2 = h*h
const vol = dr^2   #particle volume
const m = rho0*vol #particle mass

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
include("distortion_MLSEvolved.jl")

#
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

#
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
    return init_velocity_multiplier*v*VECY
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
function pE_kinetic(p::Particle)::Float64
    return 0.5*m*dot(p.v, p.v)
end 

function pE_vol(p::Particle)::Float64
    return 0.5*m*c_0^2*(rho0 - p.rho)^2/p.rho^2
end

function pE_shear(p::Particle)::Float64
    G = transpose(p.A)*p.A
    return 0.25*m*c_s^2*norm(dev(G))^2
end

function pE_penalty(p::Particle)::Float64
    return 0.5*m*c_p^2*(p.lambda/rho0)^2
end


#ARGMIN FUNCTION
#---------------
function find_minimizer(f::Function, sys::ParticleSystem)::Particle
    p = sys.particles[1]
    pval = f(p)
    for q in sys.particles
          qval = f(q)
        if qval < pval
              p = q
              pval = qval
          end
    end
    return p
 end


#
#DATA SAVING
#-----------
function vec2string(a::Vector)::String
    out = ""
    for i in 1:length(a)-1
        out = out*string(a[i])*","
    end
    if length(a) > 0
        out = out*string(a[end])
    end
    out = out*"\n"
end

using Dates
function save_parameters(folder_name::String, model::DistortionModel)
    path = joinpath(folder_name, "parameters.txt")
    mkpath(folder_name)
    open(path, "w") do io
        println(io, "model = ", nameof(typeof(model)))
        println(io, "timestamp = ", Dates.now())
        println(io)
        println(io, "L = ", L)
        println(io, "W = ", W)
        println(io, "c_l = ", c_l)
        println(io, "c_s = ", c_s)
        println(io, "c_0 = ", c_0)
        println(io, "rho0 = ", rho0)
        println(io, "nu = ", nu)
        println(io, "c_p = ", c_p)
        println(io, "init_velocity_multiplier = ", init_velocity_multiplier)
        println(io, "dr = ", dr)
        println(io, "h = ", h)
        println(io, "vol = ", vol)
        println(io, "m = ", m)
        println(io, "dt = ", dt)
        println(io, "t_end = ", t_end)
        println(io, "dt_plot = ", dt_plot)
        println(io, "dt_frame = ", dt_frame)
    end
    @info "saved simulation parameters to $path"
end

#
#TIME ITERATION
#--------------------------------------------
function integration_step(model::DistortionModel, sys::ParticleSystem)
    #verlet scheme
    apply!(sys, update_v!)
    apply!(sys, update_x!)
    evolve_A!(model, sys)
    apply!(sys, update_x!)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, find_rho!)
    force_computation!(model, sys)
    apply!(sys, update_v!)
end

function main(simulation_id::String=""; model::DistortionModel=MODEL)
    println("Simulation starting")
    sys = make_geometry()
    center = find_minimizer(p -> LinearAlgebra.norm(p.x), sys)

    folder_name = isempty(simulation_id) ?
    BASE_FOLDER * string(nameof(typeof(model))) :
    BASE_FOLDER * string(nameof(typeof(model))) * "_" * simulation_id


    out = new_pvd_file("results/"*folder_name)
    csv_data = open("results/"*folder_name*"/plate.csv", "w")
    write(csv_data, 
          string("t,y,E_total,E_kinetic,E_vol,E_shear,E_penalty\n"))
    
    save_parameters("results/"*folder_name, model)

    @time for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        if (k % Int64(round(dt_plot/dt)) == 0)
            progress = string(Int(round(k/Int64(round(dt_plot/dt))))) *
            "/" * string(Int(round(Int64(round(t_end/dt))/Int64(round(dt_plot/dt)))))
            @show progress
            @show t
            y = center.x[2]
            println("N = ", length(sys.particles))
            E_kinetic = sum(p -> pE_kinetic(p), sys.particles)
            E_vol = sum(p -> pE_vol(p), sys.particles)
            E_shear = sum(p -> pE_shear(p), sys.particles)
            E_penalty = sum(p -> pE_penalty(p), sys.particles)
            E_total = E_kinetic + E_vol + E_shear + E_penalty
            @show E_total
            @show E_kinetic
            @show E_vol
            @show E_shear
            @show E_penalty
            println()
            write(csv_data, vec2string([t, y, E_total, E_kinetic,
                                        E_vol, E_shear, E_penalty]))
        end
        if (k % Int64(round(dt_frame/dt)) == 0)
            save_frame!(out, sys, :v, :A, :e, :P, :Q, :invQ, :L,
                        :eQ, :error, :x_avg,
                        :pressure, :dev_stress, :norm_devG, :G,
                        :rho, :C_rho, :lambda, 
                        :C_lambda, :ker_sum,
                        :E_kinet, :E_shear, :E_vol, :E_penalty, 
                        :f, :condP, :condQ)
        end
        integration_step(model, sys)
    end

    save_pvd_file(out)
    close(csv_data)
end

#
# Extension of apply operator for ternary functions
#--------------------------------------------------
import SmoothedParticles: AbstractParticle, ParticleSystem

function dist2(p::AbstractParticle, q::AbstractParticle)::Float64
    return dot(p.x - q.x, p.x - q.x)
end 

@inline function _apply_ternary!(sys::ParticleSystem, action!::Function, p::AbstractParticle)
    key = SmoothedParticles.find_key(sys, p.x)
    # collect neighbours of p
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

end #module
