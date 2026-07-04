using CSV
using DataFrames
using Plots

function make_plots(filename::String)
    df = CSV.read(filename, DataFrame)
    println("Columns detected: ", names(df))

    required = ["t", "y", "E_total", "E_kinetic",
                "E_vol", "E_shear", "E_penalty"]
    for col in required
        if !(col in names(df))
            error("Missing column: $col")
        end
    end

    # Time and Y-position
    t = df[!,"t"]
    y = df[!,"y"]

    # Normalize energies
    E0 = df[!,"E_total"][3]
    E_total   = df[!,"E_total"]   ./ E0
    E_kinetic = df[!,"E_kinetic"] ./ E0
    E_vol    = df[!,"E_vol"]    ./ E0
    E_shear   = df[!,"E_shear"]   ./ E0
    E_penalty = df[!,"E_penalty"] ./ E0

    # --- Plot 1: Y Position of beam tip ---
    p1 = plot(t, y,
        xlabel = "Time [s]",
        ylabel = "Y Position",
        title = "Plate center Y Position Over Time",
        lw = 2,
        color = :purple,
        legend = false,
        grid = true
    )

    # --- Plot 2: Normalized Energies ---
    p2 = plot(t, E_total, label="Total", lw=2.5, color=:black)
    plot!(t, E_kinetic, label="Kinetic", lw=2, color=:blue)
    plot!(t, E_vol, label="Vol", lw=2, color=:green)
    plot!(t, E_shear, label="Shear", lw=2, color=:red)
    plot!(t, E_penalty, label="Penalty", lw=2, color=:orange)
    xlabel!("Time [s]")
    ylabel!("Energy / E₀")
    title!("Normalized Energy Components")
    # grid!(true)

    # --- Combine both plots vertically ---
    #plot(p1, p2, layout = (2, 1), size=(900, 700))
    plot(p2)
end


