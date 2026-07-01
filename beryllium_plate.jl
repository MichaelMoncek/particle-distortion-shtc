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
#CONSTANT PARAMETERS
#-------------------------------
const folder_name = "final_report/evolved_A_double_00"

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
# @inbounds function det(A::RealMatrix)::Float64
#     return A[1,1]*A[2,2] - A[1,2]*A[2,1]
# end
#
# @inbounds function inv(A::RealMatrix)::RealMatrix
#     idet = 1.0/det(A)
#     return RealMatrix(
#         +idet*A[2,2], -idet*A[2,1], 0., 
#         -idet*A[1,2], +idet*A[1,1], 0.,
#         0., 0., 0.
#     )
# end
#
# @inbounds function trans(A::RealMatrix)::RealMatrix
#     return RealMatrix(
#         A[1,1], A[1,2], 0., 
#         A[2,1], A[2,2], 0.,
#         0.,  0., 0.
#     )
# end
#
# @inbounds function dev(G::RealMatrix)::RealMatrix
#     λ = 1/2*(G[1,1] + G[2,2])# + 1.0)
#     return RealMatrix(
#         G[1,1] - λ,G[2,1], 0.0,
#         G[1,2], G[2,2] - λ, 0.0,
#         0.0, 0.0, 0.0#1.0 - λ
#     )
# end
#
#DEFINE VARIABLES
#------------------------------

@with_kw mutable struct Particle <: AbstractParticle
	x::RealVector         #position
    x_avg::RealVector = VEC0
    v::RealVector = init_velocity(x) #velocity
    x_old::RealVector = VEC0 #transport velocity
    f::RealVector = VEC0  #force
    X::RealVector = x     #Lag. position
    X_avg::RealVector = VEC0     #mean Lag. position
    A::FlatMatrix = FMAT1  #distortion
    A_new::FlatMatrix = FMAT1  #new distortion
    A_double_sum::FlatMatrix = FMAT1  #new distortion
    e::Float64 = 0.       #fronorm squared of eta
    rho::Float64 = 0.0    #density
    C_rho::Float64 = 0.0
    lambda::Float64 = 0.0
    C_lambda::Float64 = 0.0
    ker_sum::Float64 = 0.0
    L::FlatMatrix = FMAT1 #velocity gradient
    L_double::FlatMatrix = FMAT1 #velocity gradient
    # neigh::Float64 = 0.0#h/dr
    # Omega::Float64 = 0.0
    # h::Float64 = h
    # Diagnostics
    error::RealVector = VEC0
    f_corr::RealVector = VEC0
    f_lin::RealVector = VEC0
    f_shear::RealVector = VEC0
    f_shear_numeric::RealVector = VEC0
    f_shear_error::RealVector = VEC0
    f_constraint::RealVector = VEC0 
    P::FlatMatrix = FMAT0
    Q::FlatMatrix = FMAT0
    P_new::FlatMatrix = FMAT0
    Q_new::FlatMatrix = FMAT0
    P_double_sum::FlatMatrix = FMAT0
    Q_double_sum::FlatMatrix = FMAT0
    A_error::Float64 = 0.0
    A_single_double_sum_error::Float64 = 0.0
    P_old::FlatMatrix = FMAT0
    Q_old::FlatMatrix = FMAT0
    condP::Float64 = 0.0
    condQ::Float64 = 0.0
    condQ_new::Float64 = 0.0
    condQ_double_sum::Float64 = 0.0
    invQ::FlatMatrix = FMAT0
    eQ::Float64 = 0.0
    pressure::Float64 = 0.0 
    Pi::FlatMatrix = FMAT0
    S::FlatMatrix = FMAT0
    # gradC::RealVector = VEC0
    # v_shift::RealVector = VEC0
    # v_shift_fick::RealVector = VEC0
    # Energies
    E_kinet::Float64 = 0.0
    E_shear::Float64 = 0.0
    E_vol::Float64 = 0.0
    E_penalty::Float64 = 0.0
    G::FlatMatrix = FMAT0
    norm_devG::Float64 = 0.0
end

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
    # force_computation_evolved!(sys, 0.)
    force_computation_evolved_double!(sys, 0.)
    # force_computation!(sys, 0.)
    # force_computation_new!(sys, 0.)
    # force_computation_double!(sys, 0.)

    return sys
end



# Force computation functions
#---------------------------------------------------------------

function force_computation!(sys::ParticleSystem, t::Float64)
    apply!(sys, find_A!)
    # apply!(sys, find_A_new!)
    # apply_ternary!(sys, find_A_new_double_sum!)
    # apply!(sys, find_Pi!)
    apply!(sys, find_stress!)
    apply!(sys, find_e!)
    # apply!(sys, find_f!)
    apply!(sys, find_force!)
end

function force_computation_evolved!(sys::ParticleSystem, t::Float64)
    apply!(sys, find_L!)
    apply!(sys, find_stress_evolved!)
    apply!(sys, find_e!)
    # apply!(sys, find_f!)
    apply!(sys, find_force_evolved!)
end

function force_computation_evolved_double!(sys::ParticleSystem, t::Float64)
    apply_ternary!(sys, find_L_double!)
    apply!(sys, find_stress_evolved_double!)
    apply!(sys, find_e!)
    # apply!(sys, find_f!)
    apply!(sys, find_bulk_force_evolved_double!)
    apply_ternary!(sys, find_shear_force_evolved_double!)
end

function force_computation_new!(sys::ParticleSystem, t::Float64)
    # apply!(sys, find_A!)
    apply!(sys, find_A_new!)
    # apply_ternary!(sys, find_A_new_double_sum!)
    # apply!(sys, find_Pi!)
    apply!(sys, find_stress_new!)
    apply!(sys, find_e_new!)
    # apply!(sys, find_f!)
    apply!(sys, find_force_new!)
end

function force_computation_double!(sys::ParticleSystem, t::Float64)
    # apply!(sys, find_A!)
    # apply!(sys, find_A_new!)
    apply_ternary!(sys, find_A_new_double_sum!)
    # apply!(sys, find_Pi!)
    apply!(sys, find_stress_double!)
    # apply!(sys, find_e_new!)
    # apply!(sys, find_f!)
    # apply!(sys, find_bulk_force_double!)
    # apply_ternary!(sys, find_shear_force_double!)
end

function init_velocity(x::RealVector)::RealVector
    A = 4.3369e-5
    omega = 2.3597e5
    alpha = 78.834
    a1 = 56.6368
    a2 = 57.6455
    s = alpha*(x[1] + L/2)
    v = A*omega*(a1*(sinh(s) + sin(s)) - a2*(cosh(s) + cos(s)))
    return 0.0*1*v*VECY
end


# Different formulations of Distortion field A
# ----------------------------------------------------

# This is the single sum definition of distortion field A.
# It is simplified version of the Hutter-Pavelka definiton
function find_A!(p::Particle, q::Particle, r::Float64)
    ker = wendland2(h,r)
    x_pq = p.x - q.x
    X_pq = p.X - q.X
    p.P += ker*outer(X_pq, x_pq)
    p.Q += ker*outer(x_pq, x_pq)

end

# This is the velocity gradient arising from 
# the single sum definition of distortion field A
function find_L!(p::Particle, q::Particle, r::Float64)
    ker = wendland2(h,r)
    # velocity gradient L
    v_pq = p.v - q.v
    x_pq = p.x - q.x
    p.L += ker*outer(v_pq, x_pq)
    p.Q += ker*outer(x_pq, x_pq)
end 

# Midpoint rule for A
function update_A!(p::Particle)
    p.L = p.L*inv(p.Q)
    p.A = p.A*(FMAT1 - 0.5*dt*p.L)*inv(FMAT1 + 0.5*dt*p.L)
    # p.A = p.A*(FMAT1 - 0.5*dt*p.L)*StaticArrays.inv(FMAT1 + 0.5*dt*p.L)
end

# This is the Pavelka-Hutter definiton rewritten in a slightly 
# different way which utilizes only one sum and therefore reduces
# computational costs significantly.
function find_A_new!(p::Particle, q::Particle, r::Float64)
    ker = wendland2(h,r)
    # y_q = q.x - p.x_avg
    # Y_q = q.X - p.X_avg
    y_q = (q.x - p.x) - p.x_avg  # everything relative to p.x
    Y_q = (q.X - p.X) - p.X_avg
    p.P_new += ker*outer(Y_q, y_q)
    p.Q_new += ker*outer(y_q, y_q)
end

# This is the new particle based definition of A as
# defined in Pavelka-Hutter. 
# Note that this definition is very impractical as we need to 
# sum over two sets of indeces, which makes it very slow.
function find_A_new_double_sum!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64) 
    ker = wendland2(h,rq)*wendland2(h,rr) 
    x_qr = q.x - r.x 
    X_qr = q.X - r.X 
    p.P_double_sum += ker*outer(X_qr, x_qr) 
    p.Q_double_sum += ker*outer(x_qr, x_qr)
    # p.P_new += ker*outer(X_qr, x_qr) 
    # p.Q_new += ker*outer(x_qr, x_qr)
end 


# This is the velocity gradient arising from 
# the double sum (Pavelka-Hutter) definition of distortion field A
function find_L_double!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64)
    ker = wendland2(h,rq)*wendland2(h, rr)
    # velocity gradient L
    v_qr = q.v - r.v
    x_qr = q.x - r.x
    p.L_double += ker*outer(v_qr, x_qr)
    p.Q_double_sum += ker*outer(x_qr, x_qr)
end 

# Midpoint rule for A_double
function update_A_double!(p::Particle)
    p.L_double = p.L_double*inv(p.Q_double_sum)
    p.A_double_sum = p.A_double_sum*(FMAT1 - 0.5*dt*p.L_double)*inv(FMAT1 + 0.5*dt*p.L_double)
    # p.A = p.A*(FMAT1 - 0.5*dt*p.L)*StaticArrays.inv(FMAT1 + 0.5*dt*p.L)
end


#PHYSICS
#-------------------------------------

function find_rho!(p::Particle, q::Particle, r::Float64) 
    p.rho += m*wendland2(h,r)
    p.lambda+= m*wendland2h(h,r)
    # p.Omega += - 0.5 * r^2 * rDwendland2(p.h,r)

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


# find_stress functions
# --------------------------------------------------------------

function find_Pi!(p::Particle)
    # Shephard correction
    # p.rho = p.rho / (p.neigh*vol)
    p.rho += p.C_rho
    p.lambda += p.C_lambda 
    invQ = inv(p.Q) 
    p.invQ = invQ
    p.A = p.P*invQ
    # p.A = FMAT1
 
    # get Hutter-Pavelka A using single sum
    # get A_new and the norm of A_new - A
    invQ_new = inv(p.Q_new)
    p.invQ = invQ_new
    p.A_new = p.P_new*invQ_new
    p.A_error = norm(p.A - p.A_new)
    #
    # # get Hutter-Pavelka A using double sum
    # invQ_double_sum = inv(p.Q_double_sum)
    # p.A_double_sum = p.P_double_sum*invQ_double_sum
    # p.A_single_double_sum_error = norm(p.A_double_sum - p.A_new)

    # p.A = p.A_new
    # p.Q = p.Q_new
    # invQ = inv(p.Q)

    At = trans(p.A)
    G = At*p.A
    devG = dev(G)
    p.G = G
    p.norm_devG = norm(devG)
    # Π_a = c_s² * A^T*A * dev(A^T*A) * Q^{-T}   [Eq. 15]
    # p.Pi = c_s^2*G*devG*trans(invQ)
    #  Q is symmetric
    # p.S = p.A*devG*trans(invQ)
    # p.Pi = G*devG*trans(invQ)
    p.S = p.A*devG*invQ
    p.Pi = G*devG*invQ
    # Π'_a = c_s² * Q^{-1} * dev(A^T*A) * A^T
    # p.PiCp = c_s^2*invQ*dev(G)*At
    # P = ρ² * ε_ρ = c_0² * ρ_0 * (1 - ρ_0/ρ)
    # ill-conditioned formula 
    # pressure =  c_0^2*rho0*(1.0 - rho0/p.rho)
    # pressure =  (c_0/p.rho)^2*(rho0/p.rho)*(p.rho - rho0)/p.rho
    pressure =  c_0^2*(rho0/p.rho)*(p.rho - rho0)/p.rho
    p.pressure = pressure
    # p.pressure = pressure/p.rho^2
    # Diagnostics
    p.condQ = LinearAlgebra.cond(p.Q)#LinearAlgebra.cond(p.Q[1:2,1:2])
    p.condP = LinearAlgebra.cond(p.P)#LinearAlgebra.cond(p.P[1:2,1:2])
    p.condQ_new = LinearAlgebra.cond(p.Q_new)
    p.condQ_double_sum = LinearAlgebra.cond(p.Q_double_sum)
end

function find_stress!(p::Particle)
    p.rho += p.C_rho
    invQ = inv(p.Q) 
    p.invQ = invQ
    p.A = p.P*invQ
    At = trans(p.A)

    G = At*p.A
    devG = dev(G)
    p.G = G
    p.norm_devG = norm(devG)
    p.Pi = G*devG*invQ
    pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
    p.pressure = pressure
    # Diagnostics
    p.condQ = LinearAlgebra.cond(p.Q)#LinearAlgebra.cond(p.Q[1:2,1:2])
    p.condP = LinearAlgebra.cond(p.P)#LinearAlgebra.cond(p.P[1:2,1:2])
    # p.condQ_new = LinearAlgebra.cond(p.Q_new)
    # p.condQ_double_sum = LinearAlgebra.cond(p.Q_double_sum)
end
#

function find_stress_evolved!(p::Particle)
    p.rho += p.C_rho
    p.lambda += p.C_lambda
    At = trans(p.A)
    G = At*p.A
    devG = dev(G)
    p.G = G
    p.norm_devG = norm(devG)
    invQ = inv(p.Q)
    p.Pi = G*devG*invQ
    pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
    p.pressure = pressure
end

function find_stress_evolved_double!(p::Particle)
    p.rho += p.C_rho
    p.lambda += p.C_lambda
    At = trans(p.A_double_sum)
    G = At*p.A_double_sum
    devG = dev(G)
    p.G = G
    p.norm_devG = norm(devG)
    invQ = inv(p.Q_double_sum)
    p.Pi = G*devG*invQ
    pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
    p.pressure = pressure
end

function find_stress_new!(p::Particle)
    p.rho += p.C_rho
    # p.ker_sum += p.C_lambda 
    invQ = inv(p.Q) 
    p.invQ = invQ
    p.A = p.P*invQ
    # p.A = FMAT1
 
    # get Hutter-Pavelka A using single sum
    # get A_new and the norm of A_new - A
    invQ_new = inv(p.Q_new)
    p.A_new = p.P_new*invQ_new
    p.A_error = norm(p.A - p.A_new)
    
    # # get Hutter-Pavelka A using double sum
    # invQ_double_sum = inv(p.Q_double_sum)
    # p.A_double_sum = p.P_double_sum*invQ_double_sum
    # p.A_single_double_sum_error = norm(p.A_double_sum - p.A_new)

    # p.A = p.A_new
    # p.Q = p.Q_new
    # invQ = inv(p.Q)

    At_new = trans(p.A_new)
    G_new = At_new*p.A_new
    devG_new = dev(G_new)
    p.G = G_new
    p.norm_devG = norm(devG_new)
    # Π_a = c_s² * A^T*A * dev(A^T*A) * Q^{-T}   [Eq. 15]
    # p.Pi = c_s^2*G*devG*trans(invQ)
    #  Q is symmetric
    # p.S = p.A*devG*trans(invQ)
    # p.Pi = G*devG*trans(invQ)
    p.S = p.A_new*devG_new*invQ_new
    p.Pi = G_new*devG_new*invQ_new
    # Π'_a = c_s² * Q^{-1} * dev(A^T*A) * A^T
    # p.PiCp = c_s^2*invQ*dev(G)*At
    # P = ρ² * ε_ρ = c_0² * ρ_0 * (1 - ρ_0/ρ)
    # ill-conditioned formula 
    # pressure =  c_0^2*rho0*(1.0 - rho0/p.rho)
    # pressure =  (c_0/p.rho)^2*(rho0/p.rho)*(p.rho - rho0)/p.rho
    # pressure =  c_0^2*(rho0/p.rho)*(p.rho - rho0)/p.rho
    pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
    p.pressure = pressure
    # p.pressure = pressure/p.rho^2
    # Diagnostics


    p.condQ = LinearAlgebra.cond(p.Q)#LinearAlgebra.cond(p.Q[1:2,1:2])
    p.condP = LinearAlgebra.cond(p.P)#LinearAlgebra.cond(p.P[1:2,1:2])
    p.condQ_new = LinearAlgebra.cond(p.Q_new)
    # p.condQ_double_sum = LinearAlgebra.cond(p.Q_double_sum)
end


function find_stress_double!(p::Particle)
    p.rho += p.C_rho
    # p.ker_sum += p.C_lambda 
    invQ = inv(p.Q) 
    p.invQ = invQ
    p.A = p.P*invQ
    # p.A = FMAT1
 

    # get Hutter-Pavelka A using single sum
    # get A_new and the norm of A_new - A
    # invQ_new = inv(p.Q_new)
    # p.A_new = p.P_new*invQ_new
    # p.A_error = norm(p.A - p.A_new)
    
    # # get Hutter-Pavelka A using double sum
    invQ_double_sum = inv(p.Q_double_sum)
    p.A_double_sum = p.P_double_sum*invQ_double_sum
    # p.A_single_double_sum_error = norm(p.A_double_sum - p.A_new)

    p.A_new = p.A_double_sum
    p.P_new = p.P_double_sum
    p.Q_new = p.Q_double_sum
    invQ_new = inv(p.Q_double_sum)

    # assemble velocity gradient L_double
    # p.L_double = p.L_double * invQ_new

    At_new = trans(p.A_new)
    G_new = At_new*p.A_new
    devG_new = dev(G_new)
    p.G = G_new
    p.norm_devG = norm(devG_new)
    # Π_a = c_s² * A^T*A * dev(A^T*A) * Q^{-T}   [Eq. 15]
    # p.Pi = c_s^2*G*devG*trans(invQ)
    #  Q is symmetric
    # p.S = p.A*devG*trans(invQ)
    # p.Pi = G*devG*trans(invQ)
    p.S = p.A_new*devG_new*invQ_new
    p.Pi = G_new*devG_new*invQ_new
    # Π'_a = c_s² * Q^{-1} * dev(A^T*A) * A^T
    # p.PiCp = c_s^2*invQ*dev(G)*At
    # P = ρ² * ε_ρ = c_0² * ρ_0 * (1 - ρ_0/ρ)
    # ill-conditioned formula 
    # pressure =  c_0^2*rho0*(1.0 - rho0/p.rho)
    # pressure =  (c_0/p.rho)^2*(rho0/p.rho)*(p.rho - rho0)/p.rho
    # pressure =  c_0^2*(rho0/p.rho)*(p.rho - rho0)/p.rho
    pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
    p.pressure = pressure
    # p.pressure = pressure/p.rho^2
    # Diagnostics


    p.condQ = LinearAlgebra.cond(p.Q)#LinearAlgebra.cond(p.Q[1:2,1:2])
    p.condP = LinearAlgebra.cond(p.P)#LinearAlgebra.cond(p.P[1:2,1:2])
    p.condQ_new = LinearAlgebra.cond(p.Q_new)
    # p.condQ_double_sum = LinearAlgebra.cond(p.Q_double_sum)
end
# function find_h!(p::Particle)
#     # finalize Omega
#     p.Omega = 1.0 + p.Omega/p.neigh
#     # Newton iteration for h
#     h_new = p.h
#     for _ in 1:5 
#         h_new = h_new - (h_new - eta/sqrt(p.neigh)) / p.Omega
#     end
#     p.h = clamp(h_new, 0.5h, 3.0h)
# end

function find_force!(p::Particle, q::Particle, r::Float64) 
    ker = wendland2(h, r)
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x

    # force_shear_p = m*c_s^2*ker*p.Pi/m*x_pq
    force_shear_p = m*c_s^2*ker*p.Pi*x_pq
    p.f += force_shear_p
    # force_shear_p = m*c_s^2*ker*p.Pi/m*x_pq
    force_shear_q = m*c_s^2*ker*q.Pi*x_pq
    p.f += force_shear_q

    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
    
    # The additional force component of order o(e_pq) = o(inv(p.A)*X_pq - x_pq)
    force_geom_p = ker*m*trans(p.Pi)*p.error
    force_geom_p += rDker*m*dot(p.error, p.Pi*x_pq)*x_pq

    force_geom_q = ker*m*trans(q.Pi)*q.error
    force_geom_q += rDker*m*dot(q.error, q.Pi*x_pq)*x_pq

    p.f += force_geom_p + force_geom_q
end

function find_force_evolved!(p::Particle, q::Particle, r::Float64) 
    ker = wendland2(h, r)
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    kerh = rDwendland2h(h,r)
    # force_shear_p = m*c_s^2*ker*p.Pi/m*x_pq
    force_shear_p = m*c_s^2*ker*p.Pi*x_pq
    p.f += force_shear_p
    # force_shear_p = m*c_s^2*ker*p.Pi/m*x_pq
    force_shear_q = m*c_s^2*ker*q.Pi*x_pq
    p.f += force_shear_q

    # p.f = VEC0
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq

    # anti-clumping force 
    # kerh = rDwendland2h(h,r)
    # kerh = m/rho0*rDwendland2h(h,r)
    kerh = m/p.rho*rDwendland2h(h,r)
    # p.f += -m*kerh*(c_p/rho0)^2*(p.lambda + q.lambda)*x_pq
    p.f += -m*kerh*c_p^2*(p.lambda + q.lambda)*x_pq
end

function find_force_new!(p::Particle, q::Particle, r::Float64) 
    ker = wendland2(h, r)
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    Wq = q.ker_sum

    # shear force acting on particle p
    force_shear_pq = m*c_s^2*ker*2*Wq*p.Pi*(p.x - q.x_avg)
    p.f += force_shear_pq

    # bulk force acting on particle p
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
end


function find_bulk_force_double!(p::Particle, q::Particle, r::Float64) 
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    # bulk force acting on particle p
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
end

function find_bulk_force_evolved_double!(p::Particle, q::Particle, r::Float64) 
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    # bulk force acting on particle p
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
end
#
# function find_shear_force_double!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64)
#     ker_pq = wendland2(h, rq)
#     ker_rq = wendland2(h, norm(q.x - r.x))
#     x_rp = r.x - p.x
#     # shear force acting on particle p
#     p.f += m*c_s^2*2*ker_pq*ker_rq*q.Pi*x_rp
# end

function find_shear_force_double!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64)
    ker_pq = wendland2(h, rq)
    ker_pr = wendland2(h, rr)
    ker_qr = wendland2(h, norm(q.x - r.x))
    x_rp = r.x - p.x
    x_qp = q.x - p.x
    # shear force acting on particle p
    p.f += m*c_s^2*ker_pq*ker_qr*r.Pi*x_rp
    p.f += m*c_s^2*ker_pr*ker_qr*q.Pi*x_qp
end

function find_shear_force_evolved_double!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64)
    ker_pq = wendland2(h, rq)
    ker_pr = wendland2(h, rr)
    ker_qr = wendland2(h, norm(q.x - r.x))
    x_rp = r.x - p.x
    x_qp = q.x - p.x
    # shear force acting on particle p
    p.f += m*c_s^2*ker_pq*ker_qr*r.Pi*x_rp
    p.f += m*c_s^2*ker_pr*ker_qr*q.Pi*x_qp
end
function find_f!(p::Particle, q::Particle, r::Float64) 
    # ker = wendland2(h, r) / (0.5*(p.neigh + q.neigh))
    # rDker = rDwendland2(h,r) / (0.5*(p.neigh + q.neigh))
    ker = wendland2(h, r)
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    #
    # # shear force: m_b * ∇w_ab * [(Π_a/m_b + Π_b/m_a)*x_ab] \cdot x_ab
    force_shear_p = m*c_s^2*rDker*dot(p.Pi*x_pq,x_pq)*x_pq
    force_shear_q = m*c_s^2*rDker*dot(q.Pi*x_pq,x_pq)*x_pq
    p.f += force_shear_p
    p.f += force_shear_q
    # p.f_shear += force_shear_p 
    # p.f_shear += force_shear_q
    #
    # # bulk force: -m_b * (ε_ρa + ε_ρb) * ∇w_ab
    # # with ε_ρ = P/ρ², P = c_0²*ρ_0*(1 - ρ_0/ρ)
    # p.f += -m^2*rDker*p.pressure*p.rho^2*x_pq
    # p.f += -m^2*rDker*q.pressure*q.rho^2*x_pq
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
    # # # geometric force
    Mp = p.Pi + trans(p.Pi)
    Mq = q.Pi + trans(q.Pi)
    force_geom_p = c_s^2*m*ker*Mp*x_pq
    force_geom_q = c_s^2*m*ker*Mq*x_pq
    p.f += force_geom_p
    p.f += force_geom_q
    # p.f_shear += force_geom_p
    # p.f_shear += force_geom_q
    #
    # # dP terms
    X_pq = p.X - q.X
    S = p.S + q.S
    St = trans(S)
    p.f += -m*c_s^2*rDker*dot(S*x_pq, X_pq)*x_pq
    p.f += -m*c_s^2*ker*St*X_pq

    # artificial viscosity
    # p.f += 2*m*vol*rDker*nu*(p.v - q.v)
    
    # anti-clumping force 
    # kerh = rDwendland2h(h,r)
    # p.f += -m*kerh*(c_p/rho0)^2*(p.lambda + q.lambda)*x_pq
end
    # # # rDker_p = rDwendland2(p.h, r)
    # rDker_q = rDwendland2(q.h, r)
    # p.f -= m^2 * p.pressure/(p.Omega*p.rho^2) * rDker_p * x_pq
    # p.f -= m^2 * q.pressure/(q.Omega*q.rho^2) * rDker_q * x_pq
    # constraint from the projected system
    # mu = 1e14
    # X_pq = p.X - q.X 
    # force_constraint = mu*wendland2(h, r)*trans(p.A)*(X_pq - p.A*x_pq)
    # p.f += force_constraint
    # p.f_constraint += force_constraint
    # O(X - Ax) correction terms (required for energy conservation)
    # X_pq = p.X - q.X
    # e_pq = X_pq - p.A*x_pq
    # e_qp = -X_pq + q.A*x_pq 
    # s_pq = m*p.PiCp*e_pq - m*q.PiCp*e_qp
    #
    # p.f -= m^2*ker/m^2*s_pq
    # p.f -= m^2*dot(s_pq, x_pq)/m^2*rDker*x_pq
    # p.f_corr += p.f - p.f_lin 
    # artificial viscosity
    # p.f += 2*m*vol*rDker*nu*(p.v - q.v)
    # p.f_corr += p.f - p.f_lin
# end

function update_v!(p::Particle)
    # p.v += 0.5*dt*p.f/m
    p.v += 0.5*dt*p.f/m
    # p.v += c_s^2*0.5*dt*p.f/m
end

function update_x!(p::Particle)
    p.x += 0.5*dt*p.v
    # p.A += -p.A * p.L
end

function reset!(p::Particle)
    # reset ALL per-step accumulators 
    p.f    = VEC0
    p.e    = 0.0
    p.P    = FMAT0
    p.Q    = FMAT0
    p.Pi   = FMAT0
    # p.invQ = MAT0
    p.eQ   = 0.0
    # p.gradC = VEC0
    p.rho  = 0.0 
    p.lambda = 0.0

    # reset the averaging values
    p.ker_sum = 0.0
    p.x_avg = VEC0
    p.X_avg = VEC0
    p.P_old = FMAT0
    p.Q_old = FMAT0
    p.P_new = FMAT0
    p.Q_new = FMAT0
    p.P_double_sum = FMAT0
    p.Q_double_sum = FMAT0
    p.L = FMAT0
    p.L_double = FMAT0
    # p.neigh = 0.0
    # p.Omega = 0.0
    # p.f_corr = VEC0
    # p.f_lin = VEC0 
    p.f_shear = VEC0
    p.f_constraint = VEC0 
end

# Diagnostics
# ------------------------------------------------
function find_e!(p::Particle, q::Particle, r::Float64)
    eta = inv(p.A)*(p.X - q.X) - (p.x - q.x)
    p.e += dot(eta, eta)
    p.error = eta
    # FI3 = RealMatrix(1.0, 0.0, 0.0,
    #                  0.0, 1.0, 0.0,
    #                  0.0, 0.0, 0.0)
    p.eQ = norm(p.Q*p.invQ - FMAT1)
end

function find_e_new!(p::Particle, q::Particle, r::Float64)
    eta = inv(p.A_new)*(p.X - q.X) - (p.x - q.x)
    p.e += dot(eta, eta)
    p.error = eta
    # FI3 = RealMatrix(1.0, 0.0, 0.0,
    #                  0.0, 1.0, 0.0,
    #                  0.0, 0.0, 0.0)
    p.eQ = norm(p.Q*p.invQ - FMAT1)
end
function particle_energy(p::Particle, N::Int, E0::Float64)
    # G = trans(p.A)*p.A
    # G0 = dev(G)
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
# function numerical_shear_force(p::AbstractParticle, sys::ParticleSystem)
#     E0 = 0.25*m*c_s^2*norm(dev(p.G))^2 
#     f_shear = p.f_shear
#     f_direction = f_shear/(norm(f_shear) + 1e-16)
#     # perturbation
#     eps = 1e-8
#     x0 = p.x
#     x1 = p.x + eps*f_direction
#     # P_old = p.P
#     # Q_old = p.Q
#     p.P = FMAT0
#     p.Q = FMAT0
#     apply!(find_A, sys)
#     p.A = p.P * inv(p.Q)
#     G_pert = trans(p.A) * p.A
#     E1 = 0.25*m*c_s^2*norm(G_pert)^2
#     force_numeric = (E1 - E0) / eps
#     p.force_numeric = force_numeric
# end
#
# function numerical_force(sys::ParticleSystem)
#     eps = 1e-12
#     for p in sys.particles 
#         p.E_shear = 0.25*m*c_s^2*norm(dev(p.G))^2
#         f_shear = p.f_shear
#         f_direction = f_shear/(norm(f_shear) + 1e-16)
#         p.x_old = copy(p.x)
#         p.x += eps*f_direction
#         p.P_old = copy(p.P)
#         p.Q_old = copy(p.Q)
#         p.P = FMAT0
#         p.Q = FMAT0
#     end
#     create_cell_list!(sys)
#     apply!(sys, find_A!)
#     for p in sys.particles
#         p.A = p.P*inv(p.Q)
#         G = trans(p.A)*p.A
#         E = 0.25*m*c_s^2*norm(dev(G))^2
#         f_shear = p.f_shear
#         f_direction = f_shear/(norm(f_shear) + 1e-16)
#         p.f_shear_numeric = (E - p.E_shear) / eps * f_direction
#         p.f_shear_error = p.f_shear - p.f_shear_numeric
#         p.x = p.x_old
#         p.P = p.P_old
#         p.Q = p.Q_old
#     end
#     create_cell_list!(sys)
# end
#
#TIME ITERATION
#--------------
function main()
    println("Simulation starting")
    # println("revise working")
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
            apply!(sys, find_e!)
            save_frame!(out, sys, :v, :A, :e, :P, :Q, :invQ, :eQ, :error, :x_avg,
                        :pressure, :Pi, :rho, :C_rho, :lambda, :C_lambda, :ker_sum,
                        :A_new, :A_error, :Q_new, :P_new, :condQ_new,
                        :L, :L_double,
                        :A_double_sum, :A_single_double_sum_error, :Q_double_sum, :P_double_sum, :condQ_double_sum,
                        :E_kinet, :E_shear, :E_vol, :E_penalty, :norm_devG, :G,
                        :f_shear, :f_shear_numeric, :f_shear_error, :f_constraint,
                        :f, :condP, :condQ)#, :dv, :h, :Omega)
        end
        #verlet scheme
        apply!(sys, update_v!)
        apply!(sys, update_x!)
        create_cell_list!(sys)
        apply!(sys, reset!)
        # apply!(sys, find_L!)        
        apply_ternary!(sys, find_L_double!)        
        # apply!(sys, update_A!)       
        apply!(sys, update_A_double!)       
        apply!(sys, update_x!)
        # apply!(sys, find_h!)
        create_cell_list!(sys)
        apply!(sys, reset!)
        apply!(sys, find_rho!)
        # apply!(sys, find_avg!)
        # force_computation_evolved!(sys, t)
        force_computation_evolved_double!(sys, t)
        # force_computation!(sys, t)
        apply!(sys, update_v!)
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
