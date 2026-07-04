# distortion_DoubleSum.jl

function find_A!(::DoubleSum, p::Particle, q::Particle,
                              r::Particle, rq::Float64, rr::Float64) 
    ker = wendland2(h,rq)*wendland2(h,rr) 
    x_qr = q.x - r.x 
    X_qr = q.X - r.X 
    p.P += ker*outer(X_qr, x_qr) 
    p.Q += ker*outer(x_qr, x_qr)
end 

find_L!(::DoubleSum, p::Particle, q::Particle,
                     r::Particle, rq::Float64, rr::Float64) = nothing 
function find_e!(::DoubleSum, p::Particle, q::Particle,
                              r::Particle, rq::Float64, rr::Float64) 
    eta = inv(p.A)*(q.X - r.X) - (q.x - r.x)
    p.e += dot(eta, eta)
    p.error = eta
    p.eQ = norm(p.Q*p.invQ - FMAT1)
end

function find_stress!(::DoubleSum, p::Particle)
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
    p.pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
    # Diagnostics
    p.condQ = LinearAlgebra.cond(p.Q)
    p.condP = LinearAlgebra.cond(p.P)
end

function find_bulk_force_double!(p::Particle, q::Particle,
                                              r::Float64) 
    rDker = rDwendland2(h,r)
    x_pq = p.x - q.x
    # bulk force acting on particle p
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
end

function find_shear_force_double!(p::Particle, q::Particle,
                     r::Particle, rq::Float64, rr::Float64)
    ker_pq = wendland2(h, rq)
    ker_pr = wendland2(h, rr)
    ker_qr = wendland2(h, norm(q.x - r.x))
    x_rp = r.x - p.x
    x_qp = q.x - p.x
    # shear force acting on particle p
    p.f += m*ker_pq*ker_qr*r.dev_stress*x_rp
    p.f += m*ker_pr*ker_qr*q.dev_stress*x_qp
end

function force_computation!(model::DoubleSum, sys::ParticleSystem)
    apply_ternary!(sys, (p,q,r, pq, pr) -> 
                   find_A!(model, p, q, r, pq, pr))
    apply!(sys, p -> find_stress!(model, p))
    apply_ternary!(sys, (p,q,r, pq, pr) -> 
                   find_e!(model, p, q, r, pq, pr))
    apply!(sys, find_bulk_force_double!)
    apply_ternary!(sys, find_shear_force_double!)
end

evolve_A!(::DoubleSum, sys::ParticleSystem) = nothing
