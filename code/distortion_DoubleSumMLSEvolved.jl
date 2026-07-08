
# distortion_DoubleSumMLSEvolved.jl

function find_A!(::DoubleSumMLSEvolved, p::Particle, q::Particle,
                r::Particle, rq::Float64, rr::Float64) 
    Vq = m / q.rho
    Vr = m / r.rho
    ker = Vq*wendland2(h,rq)*Vr*wendland2(h,rr) 
    x_qr = q.x - r.x 
    X_qr = q.X - r.X 
    p.P += ker*outer(X_qr, x_qr) 
    p.Q += ker*outer(x_qr, x_qr)
end 

function find_L!(::DoubleSumMLSEvolved, p::Particle, q::Particle,
                r::Particle, rq::Float64, rr::Float64)
    Vq = m / q.rho
    Vr = m / r.rho
    ker = Vq*wendland2(h,rq)*Vr*wendland2(h,rr) 
    # velocity gradient L
    v_qr = q.v - r.v
    x_qr = q.x - r.x
    p.L += ker*outer(v_qr, x_qr)
    p.Q += ker*outer(x_qr, x_qr)
end 

function find_e!(::DoubleSumMLSEvolved, p::Particle, q::Particle,
                              r::Particle, rq::Float64, rr::Float64) 
    eta = inv(p.A)*(q.X - r.X) - (q.x - r.x)
    p.e += dot(eta, eta)
    p.error = eta
    p.eQ = norm(p.Q*p.invQ - FMAT1)
end

function find_stress!(::DoubleSumMLSEvolved, p::Particle)
    p.rho += p.C_rho
    p.lambda += p.C_lambda
    invQ = inv(p.Q) 
    p.invQ = invQ
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
    # # anti-clumping force 
    # kerh = m/p.rho*rDwendland2h(h,r)
    # p.f += -m*kerh*c_p^2*(p.lambda + q.lambda)*x_pq
end

function find_shear_force_double!(p::Particle, q::Particle,
                     r::Particle, rq::Float64, rr::Float64)
    # Vq = m / q.rho
    # Vr = m / r.rho
    ker_pq = wendland2(h, rq)
    ker_pr = wendland2(h, rr)
    ker_qr = wendland2(h, norm(q.x - r.x))
    x_rp = r.x - p.x
    x_qp = q.x - p.x
    # shear force acting on particle p
    p.f += m*ker_pq*ker_qr*r.dev_stress*x_rp
    p.f += m*ker_pr*ker_qr*q.dev_stress*x_qp
end

function force_computation!(model::DoubleSumMLSEvolved,
                            sys::ParticleSystem)
    apply_ternary!(sys, (p,q,r, pq, pr) -> 
                   find_A!(model, p, q, r, pq, pr))
    apply!(sys, p -> find_stress!(model, p))
    apply_ternary!(sys, (p,q,r, pq, pr) -> 
                   find_e!(model, p, q, r, pq, pr))
    apply!(sys, find_bulk_force_double!)
    apply_ternary!(sys, find_shear_force_double!)
end

# Midpoint rule for A
function update_A!(::DoubleSumMLSEvolved, p::Particle)
    p.L = p.L*inv(p.Q)
    p.A = p.A*(FMAT1 - 0.5*dt*p.L)*inv(FMAT1 + 0.5*dt*p.L)
end

function evolve_A!(model::DoubleSumMLSEvolved, sys::ParticleSystem)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply_ternary!(sys, (p,q,r, pq, pr) -> 
           find_L!(model, p, q, r, pq, pr))
    apply!(sys, p -> update_A!(model, p))
end
