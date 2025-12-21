from datetime import datetime

import vessim as vs


def main():
    """Run an interactive real-time simulation with multiple microgrids."""
    clients_and_location = [
        ("hyperstack", (52.5200, 13.4050)),
        ("amd", (18.9582, 72.8311)),
    ]

    sim_start = datetime.now()
    environment = vs.Environment(sim_start=sim_start, step_size=10)

    microgrids = []
    for i, (client_name, location) in enumerate(clients_and_location):
        microgrid = environment.add_microgrid(
            name=client_name,
            actors=[
                vs.Actor(name="gpu", signal=vs.StaticSignal(200)),
                #vs.Actor(name="gpu", signal=vs.PrometheusSignal(
                #    prometheus_url="http://185.216.22.195:30826/prometheus",
                #    query=f"DCGM_FI_DEV_POWER_USAGE{{gpu=\"{i}\"}}",
                #    username="admin",
                #    password="zponvkk0HC4oi5Kn1bvI"
                #)),
                # vs.Actor(name="solar", signal=vs.Trace.load(
                #     dataset="solcast2022_global",
                #     column=client_name,
                #     params={"scale": 200, "start_time": sim_start}
                # ))
            ],
            # storage= vs.SimpleBattery(
            #     capacity=10 * i,
            #     initial_soc=0.6  # Start at 60% charge
            # ),
            grid_signals={"mci_index": vs.WatttimeSignal(
                username="logsight",
                password="f%ynjSpa8P5$5W",
                location=location,
            )},
        )
        microgrids.append(microgrid)

    environment.add_controller(vs.Monitor(microgrids, outfile="results/experiment1.csv"))
    environment.add_controller(vs.Api(microgrids, export_prometheus=True))

    environment.run(until=3600*24, rt_factor=1, behind_threshold=5)


if __name__ == "__main__":
    main()
