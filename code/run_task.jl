include("beryllium_plate.jl")

using .plate

params = plate.SimulationParameters()
params.t_end = 3e-5/100.0
# params.model = plate.DoubleSumEvolved()
params.model = plate.DoubleSumMLSEvolved()


d = plate.derived_parameters(params)
@eval plate begin
    const dr = $(d.dr)
    const h = $(d.h)
    const m = $(d.m)
    const dt= $(d.dt)
end

plate.main(params)
