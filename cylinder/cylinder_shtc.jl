#=

# Flow around cylinder

```@raw html
	<img src='../assets/cylinder.png' width="50%" height="50%" alt='missing' /><br>
```
 
```@raw html
A simulation of flow around cylinder.
All parameters of this benchmark can be found
 <a href="http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html">here.</a>
```

=#

module cylinder

using Printf
using SmoothedParticles
using Parameters
import StaticArrays
import LinearAlgebra
# include("tools.jl")

const folder_name = "results/cylinder_high_res"

#=
Declare constants
=#

#geometry parameters
const dr = 3.9e-3/2 		     #average particle distance (decrease to make finer simulation)
const vol = dr^2
const h = 2.5*dr#2.5*dr		     #size of kernel support
const chan_l = 1.0      #length of the channel
const chan_w = dr*round(0.41/dr)        #width of the channel
const cyl1 = dr*round(0.2/dr)  #x coordinate of the cylinder
const cyl2 = dr*round(0.2/dr)  #y coordinate of the cylinder
const cyl_r = 0.05           #radius of the cylinder
const wall_w = 3.0*dr#2.5*dr        #width of the wall
const inflow_l = 6.0*dr      #width of inflow layer
const outflow_l = 24.0*dr    #width of outflow layer
const outflow_x = chan_l - outflow_l



#physical parameters
const U_max = 3.0       #maximum inflow velocity
const rho0 = 1.0		#referential fluid density
const m = rho0*dr^2		#particle mass
const c = 40*U_max#40.0*U_max	#numerical speed of sound
const mu = 1.0e-3		#dynamic viscosity of water 1127 for glycerine, 1.14 for water in the original article
const nu = 0.1*h*c  	#pressure stabilization
# const P0 = 0.0   #1.2          #anti-clump term
# const c_p = 2.0#4.0#4.0#3.5   #0.1          #pressure correction term
# const alpha = 100*0.013675213675213675 #artificial viscosity
const c_shift = 0.01     #velocity shifting constant

#temporal parameters
const dt = 0.2*h/c      #time step
#const dt_frame = 0.02#0.1    #how often data is saved
const t_end = 3.0      #end of simulation
const t_acc = 0.25      #time to accelerate to full speed
# const t_start = 0.20      #time to start inflow
const dt_frame = max(dt, t_end/400)    #how often data is saved

#particle types
const FLUID = 0.
const WALL = 1.
const INFLOW = 2.
const OBSTACLE = 3.

#Reynolds number 
const Re = 2*cyl_r*U_max*2/3/mu
#=
Declare variables to be stored in a Particle
=#

mutable struct Particle <: AbstractParticle
    x::RealVector #position
    v::RealVector #velocity
    dv::RealVector #velocity shift
    a::RealVector #acceleration
    rho::Float64 #density
    Drho::Float64 #rate of density
    P::Float64 #pressure
    # lambda::Float64
    # C_lambda::Float64
    type::Float64 #particle type
    Particle(x,type) = begin
        return new(x, VEC0, VEC0, VEC0, rho0, 0., 0., type)
    end
end

function make_system()
    domain = Rectangle(-inflow_l, -10*wall_w, chan_l, chan_w + 10*wall_w)
    sys = ParticleSystem(Particle, domain, h)
    grid = Grid(dr, :square)

    #define geometry
    obstacle = Circle(cyl1, cyl2, cyl_r)
    pipe = Rectangle(-inflow_l, 0., chan_l, chan_w)
    wall = BoundaryLayer(pipe, grid, wall_w)
    wall = Specification(wall, x -> (-inflow_l <= x[1] <= chan_l))
    inflow = Specification(pipe - obstacle, x -> x[1] < 0.0)
    fluid = Specification(pipe - obstacle, x -> x[1] >= 0.0)

    #generate particles
    generate_particles!(sys, grid, fluid, x -> Particle(x, FLUID))
    generate_particles!(sys, grid, inflow, x -> Particle(x, INFLOW))
    generate_particles!(sys, grid, wall, x -> Particle(x, WALL))
    generate_particles!(sys, grid, obstacle, x -> Particle(x, OBSTACLE))
    create_cell_list!(sys)
    apply!(sys, balance_of_mass!)
    # for p in sys.particles
    #    p.C_lambda = -p.lambda
    # end
    # C_lambda_min = minimum([p.C_lambda for p in sys.particles])
    # C_lambda_wall_min = minimum([p.C_lambda for p in sys.particles if p.type == WALL])

    for p in sys.particles
        p.Drho = 0.0
        # p.lambda = 0.0
        # if p.type == INFLOW
        #     p.C_lambda = C_lambda_min
        # end
        # if p.type == WALL
        #     p.C_lambda = C_lambda_wall_min
        # end
    end
    return sys
end

#Inflow function

function set_inflow_speed!(p::Particle, t::Float64)
    if p.type == INFLOW
        s = min(1.0, t/t_acc)
        v1 = 4.0*s*U_max*p.x[2]*(chan_w - p.x[2])/chan_w^2 #+ s*0.1
        p.v = v1*VECX
        #p.v = 0*VECX
    end
end

#Define interactions between particles

@inbounds function balance_of_mass!(p::Particle, q::Particle, r::Float64)
	ker = m*rDwendland2(h,r)
	p.Drho += ker*(dot(p.x-q.x, p.v-q.v) + 2*nu*(p.rho-q.rho))
    # p.lambda += m*wendland2h(h,r)
    if p.type == FLUID && q.type == FLUID
       p.Drho += 2*nu/p.rho*(p.rho - q.rho)
    end
end

function find_pressure!(p::Particle)
    #if p.x[1] >= -inflow_l + h
    #    p.rho += p.Drho*dt
    #end    
	p.rho += p.Drho*dt
	p.Drho = 0.0
    # relax density at ouflow
    if p.type == FLUID && p.x[1] > outflow_x
        alpha = (p.x[1] - outflow_x)/outflow_l  
        p.rho = p.rho + alpha * (rho0 - p.rho)
    end
	#p.P = rho0*c^2*((p.rho/rho0)^7 - 1.0)/7 + 10
    p.P = c^2*(p.rho - rho0)*rho0/p.rho + 10 # c_0 = 40, background pressure = 10
    #p.lambda += m*wendland2h(h,r)
end

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	ker = m*rDwendland2(h,r)
	p.a += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*(p.x - q.x)
	#p.a += +2*ker*mu/rho0^2*(p.v - q.v)        #bettet incompressibility?
    p.a += +2*ker*mu/(p.rho*q.rho)*(p.v - q.v)
    #p.a += 2*alpha*h*c*ker*dot(p.v - q.v, p.x - q.x)/(r^2 + 0.01) /(p.rho+q.rho)*(p.x - q.x)
    # ker = m*rDwendland2(h/2,r)
    # p.a += -2*ker*P0/rho0^2*(p.x - q.x)
    # kerh = m*rDwendland2h(h,r)
    # p.a += -kerh*(c_p/rho0)^2*(p.lambda + q.lambda)*(p.x - q.x)
    # f = a * m
    # particle shifting 
    ker2 = wendland2(h,r)
    p.dv += 2*c_shift*vol*ker2*q.rho/((p.rho + q.rho))*(q.v-p.v)
end

function move!(p::Particle)
	p.a = VEC0
    # p.lambda = p.C_lambda
	if p.type == FLUID || p.type == INFLOW
		p.x += dt*p.v 
        p.x += dt*p.dv
        p.dv = VEC0
        if abs(p.P) > 1000          #remove particles with unphysical pressure
            # p.x += 500*VECX
            p.P = 10.0
        end
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID
		p.v += 0.5*dt*p.a
        # p.v += p.dv
	end
end

function add_new_particles!(sys::ParticleSystem)
    new_particles = Particle[]
    for p in sys.particles
        if p.type == INFLOW && p.x[1] >= 0
            p.type = FLUID
            x = p.x - inflow_l*VECX
            q = Particle(x, INFLOW)
            # q.C_lambda = p.C_lambda
            push!(new_particles, q)
        end
        # elseif (p.type == FLUID && p.x[1] < -dr)
        #     p.x[1] = 0.0            
        # end        
    end
    append!(sys.particles, new_particles)

    # create_cell_list!(sys)
    # apply!(sys, balance_of_mass!)
    # for p in sys.particles
    #    p.C_lambda = -p.lambda
    # end
    # for p in sys.particles
    #     p.Drho = 0.0
    #     p.lambda = 0.0
    # end

end

# function remove_dummy_particles(sys::ParticleSystem)
#     #new_particles = filter(p -> p.type != :DUMMY, sys.particles)
#     #return ParticleSystem(new_particles)
# end

# function add_dummy_particles(sys::ParticleSystem, length)
#     dummy_particles = Particle[]
    
#     # Create dummy particles at the end for particles at the inflow (beginning)
#     for p in sys.particles
#         if p.x[1] < 0
#             x_dummy = p.x + length * VECX
#             push!(dummy_particles, Particle(x_dummy, INFLOW))
#         end
#     end
    
#     # Create dummy particles at the inflow (beginning) for particles at the end
#     for p in sys.particles
#         if p.x[1] >= length
#             x_dummy = p.x - length * VECX
#             push!(dummy_particles, Particle(x_dummy, INFLOW))
#         end
#     end
    
#     #new_particles = vcat(sys.particles, dummy_particles)
#     #return ParticleSystem(new_particles)
# end

function calculate_force(sys::ParticleSystem)::RealVector
    F = VEC0
    for p in sys.particles
        if p.type == OBSTACLE
            F += m*p.a
        end
    end
    C = 2.0*F/((2.0*U_max/3.0)^2*(2.0*cyl_r))
    return C
end

function  main()
    sys = make_system()
	out = new_pvd_file(folder_name)
    C_D = Float64[]
    C_L = Float64[]

    #a modified Verlet scheme
	for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        apply!(sys, move!)
        add_new_particles!(sys)
        for p in sys.particles
            #if t > t_start
            set_inflow_speed!(p,t)
            #end
        end
        create_cell_list!(sys)
		apply!(sys, balance_of_mass!)
        apply!(sys, find_pressure!)
        apply!(sys, internal_force!)
        apply!(sys, accelerate!)
        #save data at selected frames
        if (k %  Int64(round(dt_frame/dt)) == 0)
            @show t           
            @show length(sys.particles)
            # save_frame!(out, sys, :v, :P, :type, :lambda, :C_lambda, :rho)
            save_frame!(out, sys, :v, :dv, :P, :type, :rho)
        end
        C = calculate_force(sys)
        push!(C_D, C[1])
        push!(C_L, C[2])
		apply!(sys, accelerate!)
	end
	save_pvd_file(out)

    println()    
    C_SPH = RealVector(sum(C_D[end-9:end]/10), sum(C_L[end-9:end]/10), 0.)
    C_exact = RealVector(5.57953523384, 0.010618948146, 0.)
    relative_error = norm(C_SPH - C_exact)/norm(C_exact)
    println("C_L")
    @show C_L
    println("C_D")
    @show C_D
    @show C_SPH
    @show C_exact
    println("dt = ", dt)   
    println("relative error = ",100*relative_error,"%")
    println("Reynolds number = ", Re)
end

end
