include("beryllium_plate.jl")

using .plate

params = plate.SimulationParameters()

d = plate.derived_parameters(params)
@eval plate begin
    const dr = $(d.dr)
    const h = $(d.h)
    const m = $(d.m)
    const dt= $(d.dt)
end

plate.main(params)
