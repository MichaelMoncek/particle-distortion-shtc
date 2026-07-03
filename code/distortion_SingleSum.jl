# distortion_SingleSum.jl

function find_A!(::SingleSum, p::Particle, q::Particle, r::Float64)
    ker = wendland2(h, r)
    x_pq = p.x - q.x
    X_pq = p.X - q.X
    p.P += ker*outer(X_pq, x_pq)
    p.Q += ker*outer(x_pq, x_pq)
end

find_L!(::SingleSum, p::Particle, q::Particle, r::Float64) = nothing 

function find_e!(::SingleSum, p::Particle, q::Particle, r::Float64)
    eta = inv(p.A)*(p.X - q.X) - (p.x - q.x)
    p.e += dot(eta, eta)
    p.error = eta
    p.eQ = norm(p.Q*p.invQ - FMAT1)
end

function find_stress!(::SingleSum, p::Particle)
    p.rho += p.C_rho
    invQ = inv(p.Q)
    p.invQ = invQ
    p.A = p.P*invQ
    At = trans(p.A)
    G = At*p.A
    devG = dev(G)
    p.G = G
    p.norm_devG = norm(devG)
    p.dev_stress = c_s^2*G*devG*invQ
    p.pressure = c_0^2*rho0*(p.rho - rho0)/p.rho^3
    p.condQ = LinearAlgebra.cond(p.Q)
    p.condP = LinearAlgebra.cond(p.P)
end

function find_force!(::SingleSum, p::Particle, q::Particle, r::Float64)
    ker = wendland2(h, r)
    rDker = rDwendland2(h, r)
    x_pq = p.x - q.x
    p.f += m*ker*p.dev_stress*x_pq
    p.f += m*ker*q.dev_stress*x_pq
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
    force_geom_p  = ker*m*trans(p.dev_stress)*p.error
    force_geom_p += rDker*m*dot(p.error, p.dev_stress*x_pq)*x_pq
    force_geom_q  = ker*m*trans(q.dev_stress)*q.error
    force_geom_q += rDker*m*dot(q.error, q.dev_stress*x_pq)*x_pq
    p.f += force_geom_p + force_geom_q
end

function force_computation!(model::SingleSum, sys::ParticleSystem)
    apply!(sys, (p,q,r) -> find_A!(model, p, q, r))
    apply!(sys, p -> find_stress!(model, p))
    apply!(sys, (p,q,r) -> find_e!(model, p, q, r))
    apply!(sys, (p,q,r) -> find_force!(model, p, q, r))
end

evolve_A!(::SingleSum, sys::ParticleSystem) = nothing
# A is rebuilt from scratch each step, nothing to integrate
