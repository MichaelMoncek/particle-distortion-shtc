# index => (dr, dt_factor, t_end, model, run_name)
const PARAM_TABLE = [
                     (dr = 0.01/40, dt_factor = 0.1, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "40_dt=0.1"),
                     (dr = 0.01/40, dt_factor = 0.05, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "40_dt=0.05"),
                     (dr = 0.01/40, dt_factor = 0.025, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "40_dt=0.025"),
                     (dr = 0.01/40, dt_factor = 0.01, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "40_dt=0.01"),
                     (dr = 0.01/40, dt_factor = 0.005, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "40_dt=0.005"),
                     (dr = 0.01/30, dt_factor = 0.1, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "30_dt=0.1"),
                     (dr = 0.01/30, dt_factor = 0.05, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "30_dt=0.05"),
                     (dr = 0.01/30, dt_factor = 0.025, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "30_dt=0.025"),
                     (dr = 0.01/30, dt_factor = 0.01, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "30_dt=0.01"),
                     (dr = 0.01/30, dt_factor = 0.005, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "30_dt=0.005"),
                     (dr = 0.01/20, dt_factor = 0.1, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "20_dt=0.1"),
                     (dr = 0.01/20, dt_factor = 0.05, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "20_dt=0.05"),
                     (dr = 0.01/20, dt_factor = 0.025, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "20_dt=0.025"),
                     (dr = 0.01/20, dt_factor = 0.01, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "20_dt=0.01"),
                     (dr = 0.01/20, dt_factor = 0.005, t_end = 3e-5/1.0, model = plate.MLSEvolved(), run_name = "20_dt=0.005"),
]
