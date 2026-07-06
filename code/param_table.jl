# index => (dr, dt_factor, t_end, model, run_name)
const PARAM_TABLE = [
                     (dr = 0.01/40, dt_factor = 0.01, t_end = 3e-5/10000, model = plate.MLSEvolved(), run_name = "00"),
                     (dr = 0.01/40, dt_factor = 0.01, t_end = 3e-5/10000, model = plate.SingleSumEvolved(), run_name = "01")

]
