# distortion_SingleSumEvolved.jl

function find_A!(::SingleSumEvolved, p::Particle, 
                                     q::Particle, r::Float64)
    ker = wendland2(h, r)
    x_pq = p.x - q.x
    X_pq = p.X - q.X
    p.P += ker*outer(X_pq, x_pq)
    p.Q += ker*outer(x_pq, x_pq)
end

function find_P!(::SingleSumEvolved, p::Particle,
                                     q::Particle, r::Float64)
    ker = wendland2(h, r)
    x_pq = p.x - q.x
    X_pq = p.X - q.X
    p.P += ker*outer(X_pq, x_pq)
end

function find_error_rel_A!(::SingleSumEvolved, p::Particle)
    A_projected = p.P*inv(p.Q)
    p.error_rel_A = norm(p.A - A_projected) / norm(A_projected)
end

function find_L!(::SingleSumEvolved, p::Particle, 
                                     q::Particle, r::Float64)
    ker = wendland2(h,r)
    # velocity gradient L
    v_pq = p.v - q.v
    x_pq = p.x - q.x
    p.L += ker * outer(v_pq, x_pq)
    p.Q += ker * outer(x_pq, x_pq)
end 

function find_e!(::SingleSumEvolved, p::Particle, 
                                     q::Particle, r::Float64)
    eta = inv(p.A)*(p.X - q.X) - (p.x - q.x)
    p.e += dot(eta, eta)
    p.error = eta
    p.eQ = norm(p.Q*p.invQ - FMAT1)
end

function find_stress!(::SingleSumEvolved, p::Particle)
    p.rho += p.C_rho
    p.lambda += p.C_lambda
    At = trans(p.A)
    G = At*p.A
    devG = dev(G)
    p.G = G
    p.norm_devG = norm(devG)
    invQ = inv(p.Q)
    p.invQ = invQ
    p.dev_stress = c_s^2*G*devG*invQ
    p.pressure =  c_0^2*rho0*(p.rho - rho0)/p.rho^3
end

function find_force!(::SingleSumEvolved, p::Particle,
                                         q::Particle, r::Float64)
    ker = wendland2(h, r)
    rDker = rDwendland2(h, r)
    x_pq = p.x - q.x
    p.f += m*ker*p.dev_stress*x_pq
    p.f += m*ker*q.dev_stress*x_pq
    p.f += -m^2*rDker*p.pressure*x_pq
    p.f += -m^2*rDker*q.pressure*x_pq
    # anti-clumping force 
    kerh = m/p.rho*rDwendland2h(h,r)
    p.f += -m*kerh*c_p^2*(p.lambda + q.lambda)*x_pq
end

function force_computation!(model::SingleSumEvolved,
                              sys::ParticleSystem)
    apply!(sys, (p,q,r) -> find_A!(model, p, q, r))
    apply!(sys, (p,q,r) -> find_L!(model, p, q, r))
    apply!(sys, p -> find_stress!(model, p))
    apply!(sys, (p,q,r) -> find_e!(model, p, q, r))
    apply!(sys, (p,q,r) -> find_force!(model, p, q, r))
end

# Midpoint rule for A
function update_A!(::SingleSumEvolved, p::Particle)
    p.L = p.L*inv(p.Q)
    p.A = p.A*(FMAT1 - 0.5*dt*p.L)*inv(FMAT1 + 0.5*dt*p.L)
end

function evolve_A!(model::SingleSumEvolved, sys::ParticleSystem)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, (p,q,r) -> find_L!(model, p, q, r))
    apply!(sys, p -> update_A!(model, p))

    # find out the relative error between recomputed A and evolved A
    apply!(sys, (p,q,r) -> find_P!(model, p, q, r))
    apply!(sys, p -> find_error_rel_A!(model, p))
end
