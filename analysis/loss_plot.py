import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    START_TIME = pd.Timestamp("2026-01-11 17:00:00+00:00")

    def loss_col(df):
        """Get the training loss column (excluding MIN/MAX)"""
        return df[next(c for c in df.columns if "train_loss" in c and "MIN" not in c and "MAX" not in c)]

    def ema(losses, alpha=0.95):
        """Compute EMA over all data points"""
        smoothed = np.zeros_like(losses)
        smoothed[0] = losses[0]
        for i in range(1, len(losses)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * losses[i]
        return smoothed

    def segment(times, raw, smoothed, gap_threshold=60):
        segments = []
        seg_start = 0
        for i in range(1, len(times)):
            if times[i] - times[i-1] > gap_threshold:
                segments.append((times[seg_start:i], raw[seg_start:i], smoothed[seg_start:i]))
                seg_start = i
        segments.append((times[seg_start:], raw[seg_start:], smoothed[seg_start:]))
        return segments

    client_dfs = [pd.read_csv(f"result/exp1/client_{i}.csv") for i in range(3)]
    return (
        START_TIME,
        client_dfs,
        ema,
        loss_col,
        mdates,
        mo,
        np,
        pd,
        plt,
        segment,
    )


@app.cell
def _(START_TIME, client_dfs, ema, loss_col, pd, plt, segment):
    _fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True)

    # MCI plot (first subplot)
    mci = pd.read_csv("mci.csv", parse_dates=["point_time"], index_col="point_time")
    mci_hours = (mci.index - START_TIME).total_seconds() / 3600
    regions = {"CAISO_NORTH": ("California", "#1f77b4"), "SPP_TX": ("Texas", "#d62728"), "NEM_SA": ("South Australia", "#2ca02c")}

    for _region, (_label, _color) in regions.items():
        _mask = mci[_region].notna()
        axes[0].step(mci_hours[_mask], mci[_region][_mask].values, color=_color, linewidth=1, label=_label)

    axes[0].axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    # axes[0].fill_between(mci_hours[mask], y1=100, color="#eee")
    axes[0].set_ylabel("gCO₂/kWh")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    # axes[0].set_ylim(0, 1000)

    # Client loss plots (subplots 2-4)
    for _idx, (_df, (_label, _color)) in enumerate(zip(client_dfs, regions.values())):
        _ax = axes[_idx + 1]
        _times = _df["time"].values.astype(float)
        _losses = loss_col(_df).values
        _smoothed = ema(_losses, 0.95)
        _segments = segment(_times, _losses, _smoothed)

        for _seg_t, _seg_raw, _seg_smooth in _segments:
            _ax.plot(_seg_t / 3600, _seg_raw, color=_color, alpha=0.2, linewidth=1)
            _ax.plot(_seg_t / 3600, _seg_smooth, color=_color, alpha=1.0, linewidth=1.5)

        _ax.set_ylabel(f"Train Loss\n{_label}")
        _ax.grid(True, linestyle='--', alpha=0.4)
        _ax.set_ylim(2.5, 5.0)

    # Inset zoom: show client_0's gap region (t≈2.2-2.5h) in client_1's subplot
    inset_ax = axes[2].inset_axes([0.08, 0.3, 0.25, 0.55])
    c0_df = client_dfs[0]
    c0_times = c0_df["time"].values.astype(float)
    c0_losses = loss_col(c0_df).values
    c0_smoothed = ema(c0_losses, 0.95)
    c0_color = list(regions.values())[0][1]
    c0_segments = segment(c0_times, c0_losses, c0_smoothed)
    for seg_t, seg_raw, seg_smooth in c0_segments:
        inset_ax.plot(seg_t / 3600, seg_raw, color=c0_color, alpha=0.2, linewidth=1)
        inset_ax.plot(seg_t / 3600, seg_smooth, color=c0_color, alpha=1.0, linewidth=1.5)
    inset_ax.set_xlim(7000 / 3600, 9000 / 3600)
    inset_ax.set_ylim(2.9, 3.5)
    inset_ax.set_title("Client 0 gap", fontsize=7, pad=2)
    inset_ax.tick_params(labelsize=6)
    inset_ax.grid(True, linestyle='--', alpha=0.3)
    axes[1].indicate_inset_zoom(inset_ax, edgecolor="black", alpha=0.6)

    # Second inset: client_0 gap at t=14000-16000s, plotted in axes[1] (right side)
    inset_ax2 = axes[1].inset_axes([0.65, 0.3, 0.25, 0.55])
    for seg_t, seg_raw, seg_smooth in c0_segments:
        inset_ax2.plot(seg_t / 3600, seg_raw, color=c0_color, alpha=0.2, linewidth=1)
        inset_ax2.plot(seg_t / 3600, seg_smooth, color=c0_color, alpha=1.0, linewidth=1.5)
    inset_ax2.set_xlim(16000 / 3600, 18000 / 3600)
    inset_ax2.set_ylim(2.8, 3.4)
    inset_ax2.set_title("Client 0 gap", fontsize=7, pad=2)
    inset_ax2.tick_params(labelsize=6)
    inset_ax2.grid(True, linestyle='--', alpha=0.3)
    axes[1].indicate_inset_zoom(inset_ax2, edgecolor="black", alpha=0.6)

    # Dual x-axis: runtime (bottom) + walltime (top)
    axes[-1].set_xlabel("Runtime (hours)")
    secax = axes[0].secondary_xaxis('top', functions=(lambda h: h, lambda h: h))
    secax.set_xlabel("Walltime")
    secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda h, _: f"{17 + int(h)}:{int((h % 1) * 60):02d}"))

    axes[-1].set_xlim(0, 15)

    plt.tight_layout()
    _fig
    return (mci,)


@app.cell
def _(START_TIME, client_dfs, ema, loss_col, mdates, pd, plt):
    one_client_df = pd.read_csv("result/baseline_1_client.csv")
    two_client_df = pd.read_csv("result/baseline_2_clients.csv")

    # Merge 3-client experiment: combine and average overlapping times
    merged = pd.concat([
        pd.DataFrame({"time": _df["time"], "loss": loss_col(_df)})
        for _df in client_dfs
    ])
    merged = merged.groupby("time").mean().reset_index().sort_values("time")

    fig, _ax = plt.subplots(figsize=(10, 6))

    def _to_real_time(df):
        return START_TIME + pd.to_timedelta(df["time"].astype(float), unit="s")

    # 1 Client baseline
    _ax.plot(_to_real_time(one_client_df), loss_col(one_client_df), color="#1f77b4", alpha=0.2, linewidth=1)
    _ax.plot(_to_real_time(one_client_df), ema(loss_col(one_client_df), 0.99), color="#1f77b4", alpha=1.0, linewidth=1, label="1 Client")

    # 2 Clients baseline
    _ax.plot(_to_real_time(two_client_df), loss_col(two_client_df), color="#ff7f0e", alpha=0.2, linewidth=1)
    _ax.plot(_to_real_time(two_client_df), ema(loss_col(two_client_df), 0.99), color="#ff7f0e", alpha=1.0, linewidth=1, label="2 Clients")

    # 3 Clients experiment (merged)
    merged_time = START_TIME + pd.to_timedelta(merged["time"].astype(float), unit="s")
    _ax.plot(merged_time, merged["loss"], color="#2ca02c", alpha=0.2, linewidth=1)
    _ax.plot(merged_time, ema(merged["loss"].values, 0.99), color="#2ca02c", alpha=1.0, linewidth=1, label="3 Clients")

    _ax.set_xlabel("Time")
    _ax.set_ylabel("Training Loss")
    _ax.set_ylim(2.5, 5.0)
    _ax.set_title("Training Loss Comparison")
    _ax.legend()
    _ax.grid(True, linestyle='--', alpha=0.6)
    _ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    _ax.axhline(y=2.67, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    fig
    return


@app.cell
def _(START_TIME, client_dfs, mci, np, pd, plt, segment):
    POWER_IDLE = 0.5
    POWER_TRAIN = 2.0

    events = pd.read_csv("result/exp1/events.csv")
    baseline_df = pd.read_csv("result/baseline_1_client.csv")

    client_regions = {"client_0": "CAISO_NORTH", "client_1": "SPP_TX", "client_2": "NEM_SA"}
    region_list = ["CAISO_NORTH", "SPP_TX", "NEM_SA"]

    prov_windows = {f"client_{i}": [] for i in range(3)}
    prov_start = {}
    for _, row in events.iterrows():
        evt, client = row["event_type"], row["client"]
        if evt == "PROVISION_COMPLETE":
            prov_start[client] = row["elapsed_s"]
        elif evt == "DEPROVISION" and client in prov_start:
            prov_windows[client].append((prov_start.pop(client), row["elapsed_s"]))
    exp_end = events["elapsed_s"].max()
    for client, start in prov_start.items():
        prov_windows[client].append((start, exp_end))

    train_segments = {}
    for _i, _df in enumerate(client_dfs):
        _t = _df["time"].values.astype(float)
        _segs = segment(_t, _t, _t)
        train_segments[f"client_{_i}"] = [(_s[0][0], _s[0][-1]) for _s in _segs]

    max_time = max(baseline_df["time"].astype(float).max(), exp_end)
    dt = 60
    grid = np.arange(0, max_time + dt, dt)

    def get_power(name):
        power = np.zeros(len(grid))
        for t0, t1 in prov_windows[name]:
            power[(grid >= t0) & (grid <= t1)] = POWER_IDLE
        for t0, t1 in train_segments[name]:
            power[(grid >= t0) & (grid <= t1)] = POWER_TRAIN
        return power

    exp_powers = {c: get_power(c) for c in client_regions}
    exp_total_power = sum(exp_powers.values())

    bl_mask = (grid >= baseline_df["time"].astype(float).min()) & (grid <= baseline_df["time"].astype(float).max())
    bl_power = np.where(bl_mask, POWER_TRAIN, 0.0)

    mci_elapsed = (mci.index - START_TIME).total_seconds().values
    mci_on_grid = {}
    for r in region_list:
        mci_on_grid[r] = np.interp(grid, mci_elapsed, mci[r].values)

    # 2-client baseline power
    bl2_df = pd.read_csv("result/baseline_2_clients.csv")
    bl2_mask = (grid >= bl2_df["time"].astype(float).min()) & (grid <= bl2_df["time"].astype(float).max())
    bl2_power = np.where(bl2_mask, POWER_TRAIN * 2, 0.0)

    exp_energy = np.sum(exp_total_power) * dt / 3600
    bl_energy = np.sum(bl_power) * dt / 3600
    bl2_energy = np.sum(bl2_power) * dt / 3600

    exp_emission_rate = sum(exp_powers[c] * mci_on_grid[r] for c, r in client_regions.items())

    region_labels = {"CAISO_NORTH": "California", "SPP_TX": "Texas", "NEM_SA": "South Australia"}
    region_colors = {"CAISO_NORTH": "#1f77b4", "SPP_TX": "#d62728", "NEM_SA": "#2ca02c"}
    bl_emission_rates = {r: bl_power * mci_on_grid[r] for r in region_list}
    bl_cum_emissions = {r: np.cumsum(bl_emission_rates[r]) * dt / 3600 / 1000 for r in region_list}
    exp_cum_emissions = np.cumsum(exp_emission_rate) * dt / 3600 / 1000

    bl_total_emissions = {r: bl_cum_emissions[r][-1] for r in region_list}
    exp_total_emissions = exp_cum_emissions[-1]

    hours = grid / 3600

    _fig, _axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    _axes[0].plot(hours, bl_power, color="gray", linestyle="--", linewidth=1.5, label="1-Client BL")
    _axes[0].plot(hours, bl2_power, color="orange", linestyle="--", linewidth=1.5, label="2-Client BL")
    _axes[0].plot(hours, exp_total_power, color="#1f77b4", linewidth=1.5, label="Experiment")
    _axes[0].set_ylabel("Power (kW)")
    _axes[0].legend(loc="upper right", fontsize=8)
    _axes[0].grid(True, linestyle="--", alpha=0.4)

    for r in region_list:
        _axes[1].plot(hours, bl_emission_rates[r], color=region_colors[r], linestyle="--", linewidth=1, label=f"BL {region_labels[r]}")
    _axes[1].plot(hours, exp_emission_rate, color="black", linewidth=1.5, label="Experiment")
    _axes[1].set_ylabel("Emissions (gCO₂/h)")
    _axes[1].legend(loc="upper right", fontsize=8)
    _axes[1].grid(True, linestyle="--", alpha=0.4)

    _axes[-1].set_xlabel("Runtime (hours)")
    _axes[-1].set_xlim(0, max_time / 3600)

    plt.tight_layout()
    _fig
    return (
        bl2_energy,
        bl_energy,
        bl_total_emissions,
        exp_energy,
        exp_total_emissions,
        region_labels,
    )


@app.cell
def _(
    bl2_energy,
    bl_energy,
    bl_total_emissions,
    exp_energy,
    exp_total_emissions,
    mo,
    region_labels,
):
    _rows = [
        ("1-Client Baseline", f"{bl_energy:.1f}", "—"),
        ("2-Client Baseline", f"{bl2_energy:.1f}", "—"),
        ("Experiment (3 Clients)", f"{exp_energy:.1f}", "—"),
    ]
    _energy_table = "| Scenario | Energy (kWh) |\n|---|---|\n"
    for _name, _e, _ in _rows:
        _energy_table += f"| {_name} | {_e} |\n"

    _carbon_table = "| Scenario | Carbon (kgCO₂) |\n|---|---|\n"
    for _r, _label in region_labels.items():
        _carbon_table += f"| 1-Client BL in {_label} | {bl_total_emissions[_r]:.2f} |\n"
    _carbon_table += f"| **Experiment (3 Clients)** | **{exp_total_emissions:.2f}** |\n"

    mo.md(f"""
    ### Energy Footprint

    {_energy_table}

    ### Carbon Footprint

    {_carbon_table}
    """)
    return


if __name__ == "__main__":
    app.run()
