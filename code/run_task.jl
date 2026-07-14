include("beryllium_plate.jl")

using .plate

params = plate.SimulationParameters()
params.t_end = 3e-5/1.0
params.dt_factor = 0.05
# params.model = plate.DoubleSumEvolved()
params.model = plate.MLSEvolved()


d = plate.derived_parameters(params)
@eval plate begin
    const dr = $(d.dr)
    const h = $(d.h)
    const m = $(d.m)
    const dt= $(d.dt)
end

plate.main(params)
