
include("beryllium_plate.jl")
using .plate
include("param_table.jl")
params = plate.SimulationParameters()

idx = parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) + 1
cfg = PARAM_TABLE[idx]

params = plate.SimulationParameters(
    model       = cfg.model,
    dr          = cfg.dr,
    dt_factor   = cfg.dt_factor,
    t_end       = cfg.t_end,
    run_name    = cfg.run_name,
)
d = plate.derived_parameters(params)
@eval plate begin
    const dr = $(d.dr)
    const h = $(d.h)
    const m = $(d.m)
    const dt= $(d.dt)
end

plate.main(params)
