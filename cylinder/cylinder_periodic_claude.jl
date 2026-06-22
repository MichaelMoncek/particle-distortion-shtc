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

#TODO: add nudging force to particles instead of pressure gradient
#TODO: add option to load simulation from set state
#TODO: fix the pressure drop issue - maybe add ghost particles add the start?

module cylinder

using Printf
using SmoothedParticles
using Parameters
import StaticArrays
import LinearAlgebra

const folder_name = "results/cylinder_periodic_1"

#=
Declare constants
=#

#geometry parameters
const dr = 3.9e-3*1.5 		     #average particle distance (decrease to make finer simulation)
const vol = dr^2
const h = 2.5*dr#2.5*dr		     #size of kernel support
const chan_l = 1.0      #length of the channel
const chan_w = dr*round(0.41/dr)        #width of the channel
const cyl1 = dr*round(0.2/dr)  #x coordinate of the cylinder
const cyl2 = dr*round(0.2/dr)  #y coordinate of the cylinder
const cyl_r = 0.05           #radius of the cylinder
const wall_w = 3.0*dr#2.5*dr        #width of the wall
const inflow_l = 3.0*dr      #width of inflow layer
const outflow_l = 3.0*dr    #width of outflow layer


#physical parameters
const U_max = 3.0       #maximum inflow velocity
const rho0 = 1.0		#referential fluid density
const m = rho0*dr^2		#particle mass
const c = 40*U_max#40.0*U_max	#numerical speed of sound
const background_pressure = 10.
const mu = 1.0e-3		#dynamic viscosity of water 1127 for glycerine, 1.14 for water in the original article
const nu = 0.1*h*c  	#pressure stabilization
# const c_p = 2.0#4.0#4.0#3.5   #0.1          #pressure correction term
# const alpha = 100*0.013675213675213675 #artificial viscosity
const c_shift = 0*0.01     #velocity shifting constant

#temporal parameters
const dt = 0.2*h/c      #time step
#const dt_frame = 0.02#0.1    #how often data is saved
const t_end = 0.1      #end of simulation
const t_acc = 0.5      #time to accelerate to full speed
const dt_frame = max(dt, t_end/50)    #how often data is saved

#particle types
const FLUID = 0.
const WALL = 1.
const OBSTACLE = 2.
const GHOST = 3.

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
    domain = Rectangle(-2*inflow_l, -10*wall_w, chan_l + 2*inflow_l,
                       chan_w + 10*wall_w)
    sys = ParticleSystem(Particle, domain, h)
    grid = Grid(dr, :square)
    #define geometry
    obstacle = Circle(cyl1, cyl2, cyl_r)
    pipe = Rectangle(-inflow_l, 0., chan_l + inflow_l, chan_w)
    wall = BoundaryLayer(pipe, grid, wall_w)
    wall = Specification(wall, x -> (0.0 <= x[1] <= chan_l))
    # inflow = Specification(pipe - obstacle, x -> x[1] < 0.0)
    fluid = Specification(pipe - obstacle, x -> (x[1] >= 0.0 && x[1] <= chan_l))
    # ghosts_left = Specification(pipe - obstacle, x -> x[1] < 0.0)
    # ghosts_right = Specification(pipe - obstacle, x -> x[1] > chan_l)
    # fluid = pipe - obstacle - ghosts_right - ghosts_left

    #generate particles
    generate_particles!(sys, grid, fluid, x -> Particle(x, FLUID))
    # generate_particles!(sys, grid, inflow, x -> Particle(x, FLUID))
    generate_particles!(sys, grid, wall, x -> Particle(x, WALL))
    generate_particles!(sys, grid, obstacle, x -> Particle(x, OBSTACLE))
    # generate_particles!(sys, grid, ghosts_left, x -> Particle(x, GHOST))
    # generate_particles!(sys, grid, ghosts_right, x -> Particle(x, GHOST))
    # create_cell_list!(sys)
    # apply!(sys, balance_of_mass!)
    # for p in sys.particles
    #    p.C_lambda = -p.lambda
    # end
    # C_lambda_min = minimum([p.C_lambda for p in sys.particles])
    # C_lambda_wall_min = minimum([p.C_lambda for p in sys.particles if p.type == WALL])

    for p in sys.particles
        p.Drho = 0.0
        # p.lambda = 0.0
        # if p.type == GHOST
        #     p.C_lambda = C_lambda_min
        # end
        # if p.type == WALL
        #     p.C_lambda = C_lambda_wall_min
        # end
    end
    return sys
end

# Poiseulle velocity profile 
function initial_velocity(y::Float64)::RealVector
    v1 = 4.0*U_max*y*(chan_w - y)/chan_w^2 #+ s*0.1
    return v1*VECX
end

# Poiseulle velocity gradient force
function body_force!(p::Particle)
    if p.type == FLUID 
        # driving pressure gradient dp/dx that produces U_max parabolic profile
        # for Poiseuille flow: dp/dx = -2*mu*U_max*8/chan_w^2
        dpdx = -8.0 * mu * U_max / chan_w^2
        p.a += RealVector(-dpdx/p.rho, 0.0, 0.0)
    end
end 

#Define interactions between particles
@inbounds function balance_of_mass!(p::Particle, q::Particle, r::Float64)
    # Ghost particles must not interact with wall/obstacle — they have no periodic mirror
    (p.type == GHOST && q.type != FLUID) && return
    (q.type == GHOST && p.type != FLUID) && return
    ker = p.rho*dr*dr*rDwendland2(h,r)
    p.Drho += ker*(dot(p.x-q.x, p.v-q.v) + 2*nu*(p.rho-q.rho))
end

function find_pressure!(p::Particle)
    # Ghosts carry copied rho/P from their source — don't let integration drift them
    p.type == GHOST && return
    p.rho += p.Drho*dt
    p.Drho = 0.0
    p.P = c^2*(p.rho - rho0)*rho0/p.rho + background_pressure
end

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
    # Ghost particles must not interact with wall/obstacle — they have no periodic mirror
    (p.type == GHOST && q.type != FLUID) && return
    (q.type == GHOST && p.type != FLUID) && return
    ker = p.rho*dr*dr*rDwendland2(h,r)
    p.a += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*(p.x - q.x)
    p.a += +2*ker*mu/(p.rho*q.rho)*(p.v - q.v)
    #p.a += 2*alpha*h*c*ker*dot(p.v - q.v, p.x - q.x)/(r^2 + 0.01) /(p.rho+q.rho)*(p.x - q.x)
        # ker = m*rDwendland2(h/2,r)
        # p.a += -2*ker*P0/rho0^2*(p.x - q.x)
        # kerh = m*rDwendland2h(h,r)
        # p.a += -kerh*(c_p/rho0)^2*(p.lambda + q.lambda)*(p.x - q.x)
        # f = a * m
        # particle shifting 
        # ker2 = wendland2(h,r)
        # p.dv += 2*c_shift*vol*ker2*q.rho/((p.rho + q.rho))*(q.v-p.v)
end

function move!(p::Particle)
    p.a = VEC0
    # p.lambda = p.C_lambda
    if p.type == FLUID 
        p.x += dt*p.v 
        # wrap x-coordinate
        if p.x[1] > chan_l 
            p.x = RealVector(p.x[1] - chan_l, p.x[2], 0.0)
            return
        end
        
        if p.x[1] < 0
            p.x = RealVector(p.x[1] + chan_l, p.x[2], 0.0)
            return 
        end
    end
end


function accelerate!(p::Particle)
	if p.type == FLUID 
		p.v += 0.5*dt*p.a
	end
end

###################################################
# Ghost particles routines                        
###################################################
# buffer zones:
# left buffer zone : x in [-inflow, 0)
# - copies of particles near the RIGHT end of domain
#
# right buffer zone : x in (chan_l, chan_l + inflow]
# - copies of particles near the LEFT end of domain
####################################################

function add_ghost_particles!(sys::ParticleSystem)
    # this might slow down the simulation quite a lot!!!
    # benchmark this!!!
    new_particles = Particle[]
    for p in sys.particles
        # wrap only FLUID particles
        p.type == FLUID || continue 

        # particles near the RIGHT boundary - 
        # copy in the LEFT buffer zone
        if p.x[1] > chan_l - inflow_l
            q = Particle(RealVector(p.x[1] - chan_l, p.x[2], 0.0), GHOST)
            q.v = p.v; q.rho = p.rho; q.P = p.P
            push!(new_particles, q)
        end

        # particles near the LEFT boundary - 
        # copy in the RIGHT buffer zone
        if p.x[1] < 0.0 + inflow_l
            q = Particle(RealVector(p.x[1] + chan_l, p.x[2], 0.0), GHOST)
            q.v = p.v; q.rho = p.rho; q.P = p.P
            push!(new_particles, q)
        end
    end
    append!(sys.particles, new_particles)
end

# Remove all particles with type GHOST from sys
function remove_ghost_particles!(sys::ParticleSystem)
    filter!(p -> p.type != GHOST, sys.particles)
end

# Force acting on obstacle calculation
function calculate_force(sys::ParticleSystem)::RealVector
    F = VEC0
    for p in sys.particles
        if p.type == OBSTACLE
            # F += m*p.a
            F += p.rho*dr*dr*p.a
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
    
    # initialize ghost particles before computing pressure
    add_ghost_particles!(sys)
    create_cell_list!(sys)
    apply!(sys, balance_of_mass!)
    apply!(sys, find_pressure!) 
    remove_ghost_particles!(sys)

    #a modified Verlet scheme
	for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        apply!(sys, move!)
        add_ghost_particles!(sys)
        create_cell_list!(sys)
		apply!(sys, balance_of_mass!)
        apply!(sys, find_pressure!)
        apply!(sys, internal_force!)
        # apply!(sys, body_force!)
        apply!(sys, accelerate!)
        #save data at selected frames
        if (k %  Int64(round(dt_frame/dt)) == 0)
            @show t           
            @show length(sys.particles)
            # save_frame!(out, sys, :v, :P, :type, :lambda, :C_lambda, :rho)
            save_frame!(out, sys, :v, :dv, :P, :type, :rho)
        end
        remove_ghost_particles!(sys)
        C = calculate_force(sys)
        push!(C_D, C[1])
        push!(C_L, C[2])
		apply!(sys, accelerate!)
	end
	save_pvd_file(out)

    println()    
    # C_SPH = RealVector(sum(C_D[end-9:end]/10), sum(C_L[end-9:end]/10), 0.)
    # C_exact = RealVector(5.57953523384, 0.010618948146, 0.)
    # # relative_error = norm(C_SPH - C_exact)/norm(C_exact)
    # println("C_L")
    # @show C_L
    # println("C_D")
    # @show C_D
    # @show C_SPH
    # @show C_exact
    println("dt = ", dt)   
    # println("relative error = ",100*relative_error,"%")
    println("Reynolds number = ", Re)
end

end


