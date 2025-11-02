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

#CONSTANT PARAMETERS
#-------------------------------

const L = 0.06   #rod length
const W = 0.01   #rod width

const pull_time = 0.5  #for how long we pull

const c_l = 9046.59   #longitudinal sound speed
const c_s = 9046.59  #shear sound speed
const c_0 = sqrt(c_l^2 + 4/3*c_s^2)  #total sound speed
const rho0 = 1845.0   #density
const nu = 1.0e-4    #artificial viscosity (surpresses noise but is not neccessary)
const c_p = 0.0*c_l  #tensile penalty term

const dr = W/40    #discretization step
const h = (10.5+10*eps(Float64))dr    #support radius
const h2 = h*h
const vol = dr^2   #particle volume
const m = rho0*vol #particle mass

const dt = 0.05dr/c_0 #time step 
const t_end = 3e-5  #total simulation time
const dt_plot = max(t_end/400, dt) #how often save txt data (cheap)
const dt_frame = max(t_end/100, dt) #how often save pvd data (expensive)

#ALGEBRAIC TOOLS
#----------------------------------------

@inbounds function outer(x::RealVector, y::RealVector)::RealMatrix
    return RealMatrix(
        x[1]*y[1], x[2]*y[1], 0., 
        x[1]*y[2], x[2]*y[2], 0.,
        0., 0., 0.
    )
end

@inbounds function det(A::RealMatrix)::Float64
    return A[1,1]*A[2,2] - A[1,2]*A[2,1]
end

@inbounds function inv(A::RealMatrix)::RealMatrix
    idet = 1.0/det(A)
    return RealMatrix(
        +idet*A[2,2], -idet*A[2,1], 0., 
        -idet*A[1,2], +idet*A[1,1], 0.,
        0., 0., 0.
    )
end

@inbounds function trans(A::RealMatrix)::RealMatrix
    return RealMatrix(
        A[1,1], A[1,2], 0., 
        A[2,1], A[2,2], 0.,
        0.,  0., 0.
    )
end

@inbounds function dev(G::RealMatrix)::RealMatrix
    λ = 1/3*(G[1,1] + G[2,2] + 1.0)
    return RealMatrix(
        G[1,1] - λ,G[2,1], 0.0,
        G[1,2], G[2,2] - λ, 0.0,
        0.0, 0.0, 1.0 - λ
    )
end

#DEFINE VARIABLES
#------------------------------

@with_kw mutable struct Particle <: AbstractParticle
	x::RealVector         #position
    v::RealVector = init_velocity(x) #velocity
    f::RealVector = VEC0  #force
    X::RealVector = x     #Lag. position
    A::RealMatrix = MAT0  #distortion
    H::RealMatrix = MAT0  #correction matrix
    B::RealMatrix = MAT0  #derivative of energy wrt A
    e::Float64 = 0.       #fronorm squared of eta
    rho::Float64 = 0.0    #density
    #C_rho::Float64 = 0.0

    P::RealMatrix = MAT0
    Q::RealMatrix = MAT0
    eps::Float64 = 0.0 
    Pi::RealMatrix = MAT0
    PiCp::RealMatrix = MAT0
    # tensile penalty variables
    lambda::Float64 = 0.0
    C_lambda::Float64 = 0.0
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
    rod = Rectangle(-L/2, -W/2, L/2, W/2)
    dom = BoundaryLayer(rod, grid, L + W)
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, rod, x -> Particle(x=x))
    create_cell_list!(sys)

    # tensile penalty initialization
    apply!(sys, find_lambda!)
    for p in sys.particles 
        p.C_lambda = -p.lambda
        #p.C_rho = rho0 - p.rho
    end

    force_computation!(sys, 0.)
    return sys
end

function force_computation!(sys::ParticleSystem, t::Float64)
    apply!(sys, find_A!)
    apply!(sys, find_lambda!)
    apply!(sys, find_Pi!)
    apply!(sys, find_f_new!)
end

function init_velocity(x::RealVector)::RealVector
    A = 4.3369e-5
    omega = 2.3597e5
    alpha = 78.834
    a1 = 56.6368
    a2 = 57.6455
    s = alpha*(x[1] + L/2)
    v = A*omega*(a1*(sinh(s) + sin(s)) - a2*(cosh(s) + cos(s)))
    return 0.1*v*VECY
end

#PHYSICS
#-------------------------------------

function find_A!(p::Particle, q::Particle, r::Float64)
    ker = wendland2(h,r)
    x_pq = p.x - q.x
    X_pq = p.X - q.X
    p.P += ker*outer(X_pq, x_pq)
    p.Q += ker*outer(x_pq, x_pq)
end

function find_lambda!(p::Particle, q::Particle, r::Float64) 
    p.lambda += m*wendland2h(h,r)
    #p.rho += m*wendland2(h,r)
end 

function find_Pi!(p::Particle)
    invQ = inv(p.Q) 
    p.A = p.P*invQ
    At = trans(p.A)
    # Pi 
    G = At*p.A
    p.Pi = c_s^2*G*dev(G)*trans(invQ)
    p.PiCp = c_s^2*invQ*dev(G)*At
    # epsilon_rho
    p.rho = rho0*det(p.A) 
    p.eps = c_0^2*rho0*(1-rho0/p.rho)/p.rho^2
end

function find_f_new!(p::Particle, q::Particle, r::Float64) 
    ker = wendland2(h, r) 
    rDker = rDwendland2(h,r) 

    x_pq = p.x - q.x
    #force
    p.f += ker*p.Pi/m*x_pq
    p.f += ker*q.Pi/m*x_pq
    #Pi = A^t*epsilon_A*Q^-t
    #Pi = c_s^2*A^t*A*dev(A^t*A)*Q^{-1}
    p.f -= rDker*p.eps*x_pq
    p.f -= rDker*q.eps*x_pq
    #Terms of size O(X-Ax) (energy will not be conserved if you remove this)
    X_pq = p.X - q.X
    e_pq = X_pq - p.A*x_pq
    e_qp = -X_pq + q.A*x_pq 
    s_pq = m*p.PiCp*e_pq - m*q.PiCp*e_qp

    p.f -= ker/m^2*s_pq
    p.f -= rDker*dot(s_pq, x_pq)/m^2*x_pq

    #anti-clumping force
    kerh = rDwendland2h(h,r)
    p.f += -m*kerh*(c_p/rho0)^2*(p.lambda + q.lambda)*x_pq

    #artificial viscosity
    p.f += 2*m*vol*rDker*nu*(p.v - q.v)
end

function update_v!(p::Particle)
    p.v += 0.5*dt*m*p.f#/m
end

function update_x!(p::Particle)
    p.x += dt*p.v
    #reset vars
    p.H = MAT0
    p.A = MAT0
    p.f = VEC0
    p.e = 0.0
    p.P = MAT0
    p.Q = MAT0
    p.lambda = 0.0
    #p.rho = 0.0
end

function find_e!(p::Particle, q::Particle, r::Float64)
    eta = inv(p.A)*(p.X - q.X) - (p.x - q.x)
    p.e += dot(eta, eta)
end

function particle_energy(p::Particle)
    d = abs(det(p.A))
    G = trans(p.A)*p.A
    G0 = dev(G)
    E_kinet = 0.5*m*dot(p.v, p.v)
    E_shear = 0.25*m*c_s^2*LinearAlgebra.norm(G0,2)^2
    E_press = m*c_l^2*(d - 1.0 - log(d))
    return E_kinet + E_shear + E_press
end 

#TIME ITERATION
#--------------
function main()
    println("Simulation starting")
    sys = make_geometry()
    out = new_pvd_file("results/plate2")
    csv_data = open("results/plate2/plate.csv", "w")

    #select top-right corner
    p_sel = argmax(p -> abs(p.x[1]) + abs(p.x[2]), sys.particles) 
    
    @time for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        if (k % Int64(round(dt_plot/dt)) == 0)
            println("t = ", k*dt)
            println("N = ", length(sys.particles))
            E = sum(p -> particle_energy(p), sys.particles)
            println("E = ", E)
            println("h = ", p_sel.x[2])
            println()
            write(csv_data, string(k*dt,",",p_sel.x[2],",",E,"\n"))
        end
        if (k % Int64(round(dt_frame/dt)) == 0)
            apply!(sys, find_e!)
            save_frame!(out, sys, :v, :A, :e, :P, :Q, 
                        :eps, :Pi, :rho, :lambda, :C_lambda)
        end
        #verlet scheme
        apply!(sys, update_v!)
        apply!(sys, update_x!)
        create_cell_list!(sys)
        force_computation!(sys, t)
        apply!(sys, update_v!)
    end
    save_pvd_file(out)
    close(csv_data)
    #plot result
    data = CSV.read("results/plate2/plate.csv", DataFrame; header=false)
    p1 = plot(data[:,1], data[:,2], label = "plate-test", xlabel = "time", ylabel = "amplitude")
    p2 = plot(data[:,1], data[:,3], label = "plate-test", xlabel = "time", ylabel = "energy")
    savefig(p1, "results/plate2/amplitude.pdf")
    savefig(p2, "results/plate2/energy.pdf")
end

end