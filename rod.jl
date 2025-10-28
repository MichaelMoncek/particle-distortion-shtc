#=

# Bending of an elastic rod

=#


module rod
using SmoothedParticles
using Parameters
using Plots
using CSV
using DataFrames
import LinearAlgebra

#CONSTANT PARAMETERS
#-------------------------------

const L = 5.0   #rod length
const W = 0.5   #rod width
const r_free = 1.0   #how much free space we want around the rod

const pull_force = 2000000.0 #pulling force [N]
const pull_time = 0.5  #for how long we pull

const c_l = 20.0   #longitudinal sound speed
const c_s = 200.0  #shear sound speed
const c_0 = sqrt(c_l^2 + 4/3*c_s^2)  #total sound speed
const rho0 = 1.0   #density
const nu = 1.0e-4    #artificial viscosity (surpresses noise but is not neccessary)

const dr = W/16    #discretization step
const h = (2.5+10*eps(Float64))dr    #support radius
const h2 = h*h
const vol = dr^2   #particle volume
const m = rho0*vol #particle mass

const dt = 0.1h/c_0 #time step 
const t_end = 1.5  #total simulation time
const dt_plot = max(t_end/400, dt) #how often save txt data (cheap)
const dt_frame = max(t_end/50, dt) #how often save pvd data (expensive)

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
    v::RealVector = VEC0  #velocity
    f::RealVector = VEC0  #force
    X::RealVector = x     #Lag. position
    A::RealMatrix = MAT0  #distortion
    H::RealMatrix = MAT0  #correction matrix
    B::RealMatrix = MAT0  #derivative of energy wrt A
    e::Float64 = 0.       #fronorm squared of eta

    P::RealMatrix = MAT0
    Q::RealMatrix = MAT0
    eps::Float64 = 0.0 
    Pi::RealMatrix = MAT0
    # detH::Float64 = 0.
    # A_new::RealMatrix = MAT0
    # H_new::RealMatrix = MAT0
    # Hi::RealMatrix = MAT0
    # Hi_new::RealMatrix = MAT0
    # A_tmp::RealMatrix = MAT0
    # A_tmp_new::RealMatrix = MAT0
    # normalizer::Float64 = 0.0
end

#CREATE INITIAL STATE
#----------------------------

function make_geometry()
    grid = Grid(dr, :square)
    rod = Rectangle(0., .0, L, W)
    dom = Rectangle(-r_free, -r_free, L + r_free, W + r_free)
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, rod, x -> Particle(x=x))
    create_cell_list!(sys)
    force_computation!(sys, 0.)
    return sys
end

# function force_computation!(sys::ParticleSystem, t::Float64)
#     apply_ternary!(sys, find_A_new!)
#     #apply!(sys, find_A!)
#     apply!(sys, find_B!)
#     apply!(sys, find_f!)
#     if t < pull_time
#         apply!(sys, pull!)
#     end
# end

function force_computation!(sys::ParticleSystem, t::Float64)
    apply!(sys, find_A!)
    apply!(sys, find_Pi!)
    apply!(sys, find_f_new!)
    if t < pull_time 
        apply!(sys, pull!)
    end 
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

# # This is the new particle based definition of A 
# function find_A_new!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64) 
#     # ker should probably be normalized
#     ker = wendland2(h,rq)*wendland2(h,rr) 
#     #p.normalizer += ker
#     x_qr = q.x - r.x 
#     X_qr = q.X - r.X 
#     p.A_new -= ker*outer(X_qr, x_qr) 
#     p.H_new -= ker*outer(x_qr, x_qr)
# end 

# function find_B!(p::Particle)
#     #Hi = inv(p.H)
#     #p.H_new /= p.normalizer
#     #p.A_new /= p.normalizer
    
#     p.A = p.A_new
#     p.H = p.H_new
#     p.detH = det(p.H)

#     Hi_new = inv(p.H_new)
#     Hi = Hi_new

#     #Store all the intermidiate matrices for debugging 
#     #p.Hi = Hi
#     p.Hi_new = Hi_new
#     p.Hi = Hi_new
    
#     p.A_tmp = p.A
#     p.A_tmp_new = p.A_new

#     #p.A = p.A*Hi
#     p.A_new = p.A_new*Hi_new
#     p.A = p.A_new

#     At = trans(p.A)
#     G = At*p.A
    
#     P = c_l^2*(det(p.A)-1.0)
#     p.B = m*(P*inv(At) + c_s^2*p.A*dev(G))*Hi
# end

# Extension of apply operator for ternary functions
#--------------------------------------------------

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

#---------------------------------------------------------

function find_f!(p::Particle, q::Particle, r::Float64)
    ker = wendland2(h,r)
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    X_pq = p.X - q.X
    #force
    p.f += -ker*(trans(p.A)*(p.B*x_pq))
    p.f += -ker*(trans(q.A)*(q.B*x_pq))
    #"eta" correction (remove this -> energy will not be conserved!)
    k_pq = +trans(p.B)*(X_pq - p.A*x_pq)
    k_qp = -trans(q.B)*(X_pq - q.A*x_pq)
    p.f += rDker*dot(x_pq, k_pq)*x_pq + ker*k_pq
    p.f -= rDker*dot(x_pq, k_qp)*x_pq + ker*k_qp
    #artificial_viscosity
    p.f += 2*m*vol*rDker*nu*(p.v - q.v)
end

function find_Pi!(p::Particle)
    invQ = inv(p.Q) 
    p.A = p.P*invQ
    # Pi 
    G = trans(p.A)*p.A
    p.Pi = c_s^2*G*dev(G)*trans(invQ)
    # epsilon_rho
    rho = rho0*det(p.A) 
    p.eps = c_0^2*rho0*(1-rho0/rho)/rho^2
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
end

function pull!(p::Particle)
    if p.X[1] > L-h
        p.f += RealVector(0., (vol*pull_force)/(h*W), 0.)
    end
end

function update_v!(p::Particle)
    p.v += 0.5*dt*m*p.f#/m
    #gravitational force
    #p.v += -VECY*0.0001*dt
    #dirichlet bc
    if p.X[1] < h
        p.v = VEC0
    end
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
    #p.A_new = MAT0
    #p.H_new = MAT0
    #p.normalizer = 0.0
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
    println("Simulatin starting")
    sys = make_geometry()
    out = new_pvd_file("results/rod")
    csv_data = open("results/rod/rod.csv", "w")

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
            save_frame!(out, sys, :v, :A, :e, :H, :P, :Q, :eps, :Pi)
        end
        #verlet scheme:
        apply!(sys, update_v!)
        apply!(sys, update_x!)
        create_cell_list!(sys)
        force_computation!(sys, t)
        apply!(sys, update_v!)
    end
    save_pvd_file(out)
    close(csv_data)
    #plot result
    data = CSV.read("results/rod/rod.csv", DataFrame; header=false)
    p1 = plot(data[:,1], data[:,2], label = "rod-test", xlabel = "time", ylabel = "amplitude")
    p2 = plot(data[:,1], data[:,3], label = "rod-test", xlabel = "time", ylabel = "energy")
    savefig(p1, "results/rod/amplitude.pdf")
    savefig(p2, "results/rod/energy.pdf")
end

end