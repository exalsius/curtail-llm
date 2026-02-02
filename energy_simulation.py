import pandas as pd
import vessim as vs

SIM_START = pd.to_datetime("2026-01-11 17:00:00+00:00")


def main():
    """Run an interactive real-time simulation with multiple microgrids."""
    clients = [
        ("client_0", "CAISO_NORTH"),
        ("client_1", "SPP_TX"),
        ("client_2", "NEM_SA"),
    ]

    environment = vs.Environment(sim_start=SIM_START, step_size=10)
    mci_data = pd.read_csv("mci.csv", index_col="point_time", parse_dates=True)

    for client_name, mci_region in clients:
        environment.add_microgrid(
            name=client_name,
            actors=[
                vs.Actor(name="gpu", signal=vs.StaticSignal(-200)),  # Consumption is negative
                # vs.Actor(name="gpu", signal=vs.PrometheusSignal(
                #    prometheus_url="http://185.216.22.195:30826/prometheus",
                #    query=f"DCGM_FI_DEV_POWER_USAGE{{gpu=\"{i}\"}}",
                #    username="admin",
                #    password="zponvkk0HC4oi5Kn1bvI"
                # )),
            ],
            grid_signals={"mci": vs.Trace(mci_data, column=mci_region)},
        )

    environment.add_controller(vs.CsvLogger(outfile="results/experiment1.csv"))
    environment.add_controller(vs.Api(export_prometheus=False, broker_port=8800))

    environment.run(until=3600*18, rt_factor=1, behind_threshold=5)


if __name__ == "__main__":
    main()
