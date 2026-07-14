#
# function make_plots(filename::String)
#     df = CSV.read(filename, DataFrame)
#     println("Columns detected: ", names(df))
#
#     required = ["t", "y", "E_total", "E_kinetic",
#                 "E_vol", "E_shear", "E_penalty"]
#     for col in required
#         if !(col in names(df))
#             error("Missing column: $col")
#         end
#     end
#
#     # Time and Y-position
#     t = df[!,"t"]
#     y = df[!,"y"]
#
#     # Normalize energies
#     E0 = df[!,"E_total"][1]
#     E_total   = df[!,"E_total"]   ./ E0
#     E_kinetic = df[!,"E_kinetic"] ./ E0
#     E_vol    = df[!,"E_vol"]    ./ E0
#     E_shear   = df[!,"E_shear"]   ./ E0
#     E_penalty = df[!,"E_penalty"] ./ E0
#
#     # --- Plot 1: Y Position of beam tip ---
#     p1 = plot(t, y,
#         xlabel = "Time [s]",
#         ylabel = "Y Position",
#         title = "Plate center Y Position Over Time",
#         lw = 2,
#         color = :purple,
#         legend = false,
#         grid = true
#     )
#
#     # --- Plot 2: Normalized Energies ---
#     p2 = plot(t, E_total, label="Total", lw=2.5, color=:black)
#     plot!(t, E_kinetic, label="Kinetic", lw=2, color=:blue)
#     plot!(t, E_vol, label="Vol", lw=2, color=:green)
#     plot!(t, E_shear, label="Shear", lw=2, color=:red)
#     plot!(t, E_penalty, label="Penalty", lw=2, color=:orange)
#     xlabel!("Time [s]")
#     ylabel!("Energy / E₀")
#     title!("Normalized Energy Components")
#     # grid!(true)
#
#     # --- Plot 3: Avg Error
#     # p3 = plot(t, 
#     # --- Plot 4: Max Error
#     # --- Combine both plots vertically ---
#     #plot(p1, p2, layout = (2, 1), size=(900, 700))
#     plot(p1)
# end
#

# all the runs live inside results/final_report
# MLSEvolved_20_dt=0.1
# MLSEvolved_20_dt=0.05
# ...
#
# each run is loaded from param_table.jl with a julia script
# slurm_run_task.jl and each run has its SimulationParameters 
# struct generated this way
#

using CSV
using DataFrames
using Plots

include("beryllium_plate.jl")
using .plate
include("param_table.jl")

function load_params(entry)
    # cfg = PARAM_TABLE[idx]
    params = plate.SimulationParameters(
        model       = entry.model,
        dr          = entry.dr,
        dt_factor   = entry.dt_factor,
        t_end       = entry.t_end,
        run_name    = entry.run_name,
    )
    return params
end


# get the assigned folder name form params
# ie (DistortionModel)_(dr)_dt=(dt)
function folder_name(params::plate.SimulationParameters)
    model = string(nameof(typeof(params.model)))
    folder_name = isempty(params.run_name) ?
        params.output_folder * model :
        params.output_folder * model * "_" * params.run_name
    return folder_name
end

# loads CSV file from path
function _load_csv(path)
    df = CSV.read(path, DataFrame)
    return df
end

# load specific run with idx
function load_run(entry)
    params = load_params(entry)
    folder = "results/" * folder_name(params)
    df = _load_csv(joinpath(folder, "plate.csv"))
    required = ["t", "y", "E_total", "E_kinetic", "E_vol",
                "E_shear", "E_penalty", 
               "total_error", "avg_error", "max_error"]

    for col in required
        if !(col in names(df))
            error("Missing column: $col in $folder/plate.csv.\n
                  Columns found: $(names(df))")
        end
    end
    return (params = params, df = df, folder = folder)
end


function load_all_runs()
    return [load_run(entry) for entry in PARAM_TABLE]
end

function select_runs(runs; dr=nothing, dt_factor=nothing)
    selected = runs
    if dr !== nothing
        selected = filter(r -> r.params.dr == dr, selected)
    end

    if dt_factor !== nothing
        selected = filter(r -> r.params.dt == dt_factor, selected)
    end

    if isempty(selected)
        error("No runs found matching dr=$(dr), dt_factor=$(dt_factor)")
    end
    return selected
end

function run_label(run)
    params = run.params
    model_name = string(nameof(typeof(params.model)))
    run_label = "$(model_name), dr=$(params.dr), dt=$(params.dt)dr/c0"
    return run_label
end

function plot_max_error(run)
    df = run.df
    plot(df.t, df.max_error,
        xlabel = "Time [s]",
        ylabel = "Max error",
        title  = "Max error over time -- " * run_label(run),
        lw     = 2,
        color  = :red,
        legend = false,
        grid   = true,
    )
end

function plot_max_error_vs_dt(runs, dt_factor)
    selected = select_runs(runs; dt_factor = dt_factor)
    selected = sort(selected, by = r -> r.params.dr)
    model_name = string(nameof(typeof(selected[1].params.model)))
    p = plot(
        xlabel = "Time [s]",
        ylabel = "Max error",
        title  = "Max error over time -- $(model_name),
        dt=$(dt_factor)dr/c0",
        legend = :topright,
        grid   = true,
    )
    for r in selected
        plot!(p, r.df.t, r.df.max_error,
              label = "dr=$(r.params.dr)", lw = 2)
    end
    return p
end

function plot_error_rel_A(run_MLS, run_SingleSum)
    p = plot(
        xlabel = "Time [s]",
        ylabel = "Relative error",
        title = "Relative error A evolved vs A recomputed",
        grid = true,)
        plot!(p, run_MLS.t, run_MLS.error_rel_A,
              label = "MLSEvolved", lw = 2)
        plot!(p, run_SingleSum.t, run_SingleSum.error_rel_A,
              label = "SingleSumEvolved", lw = 2)
        return p
end

function plot_max_error_vs_dr(runs, dr)
    selected = select_runs(runs; dr = dr)
    selected = sort(selected, by = r -> r.params.dt_factor)
    model_name = string(nameof(typeof(selected[1].params.model)))
    p = plot(
        xlabel = "Time [s]",
        ylabel = "Max error",
        title  = "Max error over time -- $(model_name), dr=$(dr)",
        legend = :topright,
        grid   = true,
    )
    for r in selected
        plot!(p, r.df.t, r.df.max_error,
              label = "dt=$(r.params.dt_factor)dr/c0", lw = 2)
    end
    return p
end

function plot_avg_error(run)
    df = run.df
    plot(df.t, df.max_error,
        xlabel = "Time [s]",
        ylabel = "Avg error",
        title  = "Avg error over time -- " * run_label(run),
        lw     = 2,
        color  = :red,
        legend = false,
        grid   = true,
    )
end

function plot_avg_error_vs_dt(runs, dt_factor)
    selected = select_runs(runs; dt_factor = dt_factor)
    selected = sort(selected, by = r -> r.params.dr)
    model_name = string(nameof(typeof(selected[1].params.model)))
    p = plot(
        xlabel = "Time [s]",
        ylabel = "Avg error",
        title  = "Avg error over time -- $(model_name),
        dt=$(dt_factor)dr/c0",
        legend = :topright,
        grid   = true,
    )
    for r in selected
        plot!(p, r.df.t, r.df.avg_error,
              label = "dr=$(r.params.dr)", lw = 2)
    end
    return p
end

function plot_avg_error_vs_dr(runs, dr)
    selected = select_runs(runs; dr = dr)
    selected = sort(selected, by = r -> r.params.dt_factor)
    model_name = string(nameof(typeof(selected[1].params.model)))
    p = plot(
        xlabel = "Time [s]",
        ylabel = "Avg error",
        title  = "Avg error over time -- $(model_name), dr=$(dr)",
        legend = :topright,
        grid   = true,
    )
    for r in selected
        plot!(p, r.df.t, r.df.avg_error,
              label = "dt=$(r.params.dt_factor)dr/c0", lw = 2)
    end
end

function plot_energies(run)
    df = run.df
    t = df[!,"t"]
    # Normalize energies
    E0 = df[!,"E_total"][1]
    E_total   = df[!,"E_total"]   ./ E0
    E_kinetic = df[!,"E_kinetic"] ./ E0
    E_vol    = df[!,"E_vol"]    ./ E0
    E_shear   = df[!,"E_shear"]   ./ E0
    E_penalty = df[!,"E_penalty"] ./ E0
    p = plot(t, E_total, label="Total", lw=2.5, color=:black)
    plot!(t, E_kinetic, label="Kinetic", lw=2, color=:blue)
    plot!(t, E_vol, label="Vol", lw=2, color=:green)
    plot!(t, E_shear, label="Shear", lw=2, color=:red)
    plot!(t, E_penalty, label="Penalty", lw=2, color=:orange)
    xlabel!("Time [s]")
    ylabel!("Energy / E₀")
    title!("Normalized Energy Components -- " * run_label(run))
    return p
end

function plot_center(run)
    df = run.df
    y = df[!,"y"]
    t = df[!,"t"]
    p = plot(t, y,
        xlabel = "Time [s]",
        ylabel = "Y Position",
        title = "Plate center Y Position -- " * run_label(run),
        lw = 2,
        color = :purple,
        legend = false,
        grid = true
    )
    return p
end


