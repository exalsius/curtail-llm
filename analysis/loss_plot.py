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
    FONTSIZE = 9

    def ema(losses, alpha=0.95):
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

    def style_time_axes(axes, xlim_hours, walltime=True):
        axes[-1].set_xlabel("Runtime (hours)", fontsize=FONTSIZE)
        axes[-1].set_xlim(0, xlim_hours)
        if walltime:
            _secax = axes[0].secondary_xaxis('top', functions=(lambda h: h, lambda h: h))
            _secax.set_xlabel("Walltime (UTC)", fontsize=FONTSIZE)
            _secax.tick_params(labelsize=FONTSIZE)
            _secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda h, _: f"{(17 + int(h)) % 24}:{int((h % 1) * 60):02d}"))
        for _ax in axes:
            _ax.tick_params(labelsize=FONTSIZE)

    COLORS = {"ours": "#ff7f0e", "bl1": "#333333", "bl2": "#999"}

    client_dfs = [pd.read_csv(f"result/exp1/client_{i}.csv") for i in range(3)]
    return (
        COLORS,
        FONTSIZE,
        START_TIME,
        client_dfs,
        ema,
        mo,
        np,
        pd,
        plt,
        segment,
        style_time_axes,
    )


@app.cell
def _(
    FONTSIZE,
    START_TIME,
    client_dfs,
    ema,
    pd,
    plt,
    segment,
    style_time_axes,
):
    HIGHLIGHT_COLOR = "#f6f8f6"


    _fig, axes = plt.subplots(4, 1, figsize=(7, 5), sharex=True)

    # MCI plot (first subplot)
    mci = pd.read_csv("mci.csv", parse_dates=["point_time"], index_col="point_time")
    mci_hours = (mci.index - START_TIME).total_seconds() / 3600
    regions = {"CAISO_NORTH": ("California", "#1f77b4"), "SPP_TX": ("Texas", "#d62728"), "NEM_SA": ("South Australia", "#2ca02c"), "DE": ("Germany", "#9467bd")}

    for _region, (_label, _color) in regions.items():
        _mask = mci[_region].notna()
        axes[0].step(mci_hours[_mask], mci[_region][_mask].values, color=_color, linewidth=1.2, label=_label)

    axes[0].axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_ylabel("carbon intensity\n(gCO₂/kWh)", fontsize=FONTSIZE)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # Client loss plots (subplots 2-4)
    for _idx, (_df, (_label, _color)) in enumerate(zip(client_dfs, regions.values())):
        _ax = axes[_idx + 1]
        _times = _df["time"].values.astype(float)
        _losses = _df["loss"].values
        _smoothed = ema(_losses, 0.95)
        _segments = segment(_times, _losses, _smoothed)

        for _seg_t, _seg_raw, _seg_smooth in _segments:
            _ax.plot(_seg_t / 3600, _seg_raw, color=_color, alpha=0.2, linewidth=1)
            _ax.plot(_seg_t / 3600, _seg_smooth, color=_color, alpha=1.0, linewidth=1.5)

        _ax.set_ylabel("train loss", fontsize=FONTSIZE)
        _ax.set_title(_label, fontsize=FONTSIZE, loc='right', y=0.7, x=0.99)
        _ax.grid(True, linestyle='--', alpha=0.4)
        _ax.set_ylim(2.5, 5.0)

    _colors = [v[1] for v in regions.values()]

    def _client_segments(idx):
        _df = client_dfs[idx]
        _t = _df["time"].values.astype(float)
        _l = _df["loss"].values
        return segment(_t, _l, ema(_l, 0.95))

    def _add_inset(*, pos, client, xlim, ylim, title, indicator_ax):
        _segs = _client_segments(client)
        _color = _colors[client]
        _inset = _fig.add_axes(pos)
        for _st, _sr, _ss in _segs:
            _inset.plot(_st / 3600, _sr, color=_color, alpha=0.2, linewidth=1)
            _inset.plot(_st / 3600, _ss, color=_color, alpha=1.0, linewidth=1.5)
        _inset.set(xlim=xlim, ylim=ylim)
        _inset.set_xlabel(title, fontsize=8)
        _inset.set_xticks([])
        _inset.set_yticks([])
        _inset.grid(True, linestyle='--', alpha=0.3)
        indicator_ax.indicate_inset_zoom(_inset, edgecolor="black", alpha=0.6)

    style_time_axes(axes, 14.65)

    plt.tight_layout()

    _add_inset(
        pos=[0.16, 0.40, 0.20, 0.10], client=0,
        xlim=(7000/3600, 9500/3600), ylim=(2.9, 3.5),
        title="training stops if no sites are\nbelow the curtailment threshold",
        indicator_ax=axes[1],
    )
    _add_inset(
        pos=[0.43, 0.40, 0.20, 0.10], client=0,
        xlim=(15500/3600, 18000/3600), ylim=(2.8, 3.4),
        title="if more than one site is cur-\ntailed, training switches to FL",
        indicator_ax=axes[1],
    )
    _add_inset(
        pos=[0.43, 0.235, 0.20, 0.10], client=2,
        xlim=(22700/3600, 25200/3600), ylim=(2.8, 3.4),
        title="",
        indicator_ax=axes[3],
    )

    _fig.subplots_adjust(hspace=0.15)
    _fig.align_labels()
    _fig.savefig("figures/loss_mci.pdf", bbox_inches="tight")
    _fig
    return (mci,)


@app.cell
def _(COLORS, FONTSIZE, client_dfs, ema, pd, plt, style_time_axes):
    _bl1 = pd.read_csv("result/baseline_1_client.csv")
    _bl2 = pd.read_csv("result/baseline_2_clients.csv")
    _ours = pd.concat([
        pd.DataFrame({"time": _df["time"], "ppl": _df["perplexity"]}) for _df in client_dfs
    ]).groupby("time").mean().reset_index().sort_values("time")

    _fig, _ax = plt.subplots(figsize=(4, 2.5))

    def _plot(ax, hours, raw, color, label):
        smoothed = ema(raw, 0.99)
        ax.plot(hours, raw, color=color, alpha=0.1, linewidth=1)
        ax.plot(hours, smoothed, color=color, linewidth=1, label=label)
        ax.plot(hours[-1], smoothed[-1], 'o', color=color, markersize=3)

    _plot(_ax, (_bl1["time"] / 3600).values, _bl1["perplexity"].values, COLORS["bl1"], "Centralized")
    _plot(_ax, (_bl2["time"] / 3600).values, _bl2["perplexity"].values, COLORS["bl2"], "Two-site FL")
    _plot(_ax, (_ours["time"] / 3600).values, _ours["ppl"].values, COLORS["ours"], "Ours")

    _ax.set_ylabel("Perplexity", fontsize=FONTSIZE)
    _ax.set_ylim(10, 50)
    _ax.legend(fontsize=FONTSIZE)
    _ax.grid(True, linestyle='--', alpha=0.6)

    _ax.axhline(y=14.7, color='gray', linestyle='--', linewidth=1, alpha=0.7)


    style_time_axes([_ax], 18, walltime=False)

    _fig.savefig("figures/perplexity.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    COLORS,
    FONTSIZE,
    START_TIME,
    client_dfs,
    mci,
    np,
    pd,
    plt,
    segment,
    style_time_axes,
):
    POWER_IDLE = 0.468
    POWER_TRAIN = 2.024

    events = pd.read_csv("result/exp1/events.csv")
    baseline_df = pd.read_csv("result/baseline_1_client.csv")

    client_regions = {"client_0": "CAISO_NORTH", "client_1": "SPP_TX", "client_2": "NEM_SA"}
    region_list = ["CAISO_NORTH", "SPP_TX", "NEM_SA", "DE"]

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

    exp_emission_rate = sum(exp_powers[c] * mci_on_grid[r] for c, r in client_regions.items())

    region_labels = {"CAISO_NORTH": "California", "SPP_TX": "Texas", "NEM_SA": "South Australia", "DE": "Germany"}
    region_colors = {"CAISO_NORTH": "#1f77b4", "SPP_TX": "#d62728", "NEM_SA": "#2ca02c", "DE": "#9467bd"}
    bl_emission_rates = {r: bl_power * mci_on_grid[r] for r in region_list}
    bl_cum_emissions = {r: np.cumsum(bl_emission_rates[r]) * dt / 3600 / 1000 for r in region_list}
    exp_cum_emissions = np.cumsum(exp_emission_rate) * dt / 3600 / 1000

    bl_total_emissions = {r: bl_cum_emissions[r][-1] for r in region_list}
    exp_total_emissions = exp_cum_emissions[-1]

    hours = grid / 3600

    _fig, _axes = plt.subplots(2, 1, figsize=(5, 3.07), sharex=True)

    _exp_mask = grid <= exp_end
    _exp_h = hours[_exp_mask]
    _exp_pow = exp_total_power[_exp_mask]
    _exp_em = (exp_emission_rate / 1000)[_exp_mask]

    _axes[0].plot(_exp_h, _exp_pow, color=COLORS["ours"], linewidth=1, label="Ours")
    _axes[0].plot(_exp_h[-1], _exp_pow[-1], 'o', color=COLORS["ours"], markersize=3)
    _axes[0].plot(hours, bl_power, color=COLORS["bl1"], linestyle="--", linewidth=1, label="Centralized")
    _axes[0].set_ylabel("Power\n(kW)", fontsize=FONTSIZE)
    _axes[0].legend(loc="upper right", fontsize=8)
    _axes[0].grid(True, linestyle="--", alpha=0.4)

    for r in region_list:
        _axes[1].plot(hours, bl_emission_rates[r] / 1000, color=region_colors[r], linestyle="--", linewidth=1, label=f"Centralized {region_labels[r]}")
    _axes[1].plot(_exp_h, _exp_em, color=COLORS["ours"], linewidth=1.5, label="Ours")
    _axes[1].plot(_exp_h[-1], _exp_em[-1], 'o', color=COLORS["ours"], markersize=3)
    _axes[1].set_ylabel("Emission rate\n(kgCO\u2082/h)", fontsize=FONTSIZE)
    _axes[1].legend(loc="upper right", fontsize=8)
    _axes[1].grid(True, linestyle="--", alpha=0.4)

    style_time_axes(_axes, max_time / 3600)

    _fig.align_labels()
    _fig.tight_layout()
    _fig.subplots_adjust(hspace=0.15)
    _fig.savefig("figures/power_emissions.pdf", bbox_inches="tight")
    _fig
    return (
        POWER_TRAIN,
        bl_emission_rates,
        bl_power,
        bl_total_emissions,
        dt,
        exp_emission_rate,
        exp_end,
        exp_total_emissions,
        exp_total_power,
        grid,
        max_time,
        region_colors,
        region_labels,
        region_list,
    )


@app.cell
def _(
    COLORS,
    FONTSIZE,
    bl_emission_rates,
    bl_total_emissions,
    exp_emission_rate,
    exp_end,
    exp_total_emissions,
    grid,
    max_time,
    plt,
    region_colors,
    region_labels,
    region_list,
    style_time_axes,
):
    _hours = grid / 3600
    _exp_mask = grid <= exp_end
    _exp_h = _hours[_exp_mask]
    _exp_em = (exp_emission_rate / 1000)[_exp_mask]

    _fig, (_ax, _ax_bar) = plt.subplots(1, 2, figsize=(8.5, 2), gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.25})

    for _r in region_list:
        _vals = bl_emission_rates[_r] / 1000
        _ax.plot(_hours, _vals, color=region_colors[_r], linestyle="--", linewidth=1, label=f"{region_labels[_r]}")
    _ax.plot(_exp_h, _exp_em, color=COLORS["ours"], linewidth=1.5, label="Ours")
    _ax.plot(_exp_h[-1], _exp_em[-1], 'o', color=COLORS["ours"], markersize=3)

    _ax.set_ylabel("Emission rate (kgCO\u2082/h)", fontsize=FONTSIZE)
    _ax.grid(True, linestyle="--", alpha=0.4)

    _ax.legend(loc="upper left", fontsize=8, ncol=3, bbox_to_anchor=(-0, 1.29), frameon=False)

    style_time_axes([_ax], max_time / 3600, walltime=False)

    _bar_labels = [region_labels[_r] for _r in region_list] + ["Ours"]
    _bar_values = [bl_total_emissions[_r] for _r in region_list] + [exp_total_emissions]
    _bar_colors = [region_colors[_r] for _r in region_list] + [COLORS["ours"]]
    _bars = _ax_bar.bar(_bar_labels, _bar_values, color=_bar_colors, width=0.7)
    for _bar, _val in zip(_bars, _bar_values):
        _ax_bar.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height(), f"{_val:.1f}", ha="center", va="bottom", fontsize=7)
    _ax_bar.set_ylabel("Total emissions (kgCO\u2082)", fontsize=FONTSIZE)
    _ax_bar.tick_params(labelsize=8, axis="x", labelrotation=30, pad=2)
    plt.setp(_ax_bar.get_xticklabels(), ha="right")
    _ax_bar.tick_params(labelsize=FONTSIZE, axis="y")
    _ax_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
    _ax_bar.set_ylim(0, max(_bar_values) * 1.1)

    # _fig.tight_layout()
    _fig.savefig("figures/emissions.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    COLORS,
    FONTSIZE,
    bl_total_emissions,
    exp_emission_rate,
    exp_end,
    exp_total_emissions,
    grid,
    plt,
    region_colors,
    region_labels,
    region_list,
):
    _hours = grid / 3600
    _exp_mask = grid <= exp_end
    _exp_h = _hours[_exp_mask]
    _exp_em = (exp_emission_rate / 1000)[_exp_mask]

    _fig, _ax_bar = plt.subplots(1, 1, figsize=(4, 2.5))

    _bar_labels = [region_labels[_r].replace(" ", "\n") for _r in region_list] + ["Ours\n(distributed)"]
    _bar_values = [bl_total_emissions[_r] for _r in region_list] + [exp_total_emissions]
    _bar_colors = [region_colors[_r] for _r in region_list] + [COLORS["ours"]]
    _bars = _ax_bar.bar(_bar_labels, _bar_values, color=_bar_colors, width=0.7)
    for _bar, _val in zip(_bars, _bar_values):
        _ax_bar.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.3, f"{_val:.1f}", ha="center", va="bottom", fontsize=9)
    _ax_bar.set_ylabel("Total emissions (kgCO\u2082)", fontsize=FONTSIZE)
    _ax_bar.tick_params(labelsize=8, axis="x")
    #plt.setp(_ax_bar.get_xticklabels(), ha="right")
    _ax_bar.tick_params(labelsize=FONTSIZE, axis="y")
    _ax_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
    _ax_bar.set_ylim(0, max(_bar_values) * 1.1)

    # _fig.tight_layout()
    _fig.savefig("figures/emissions_short.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    POWER_TRAIN,
    bl_power,
    bl_total_emissions,
    client_dfs,
    dt,
    ema,
    exp_total_emissions,
    exp_total_power,
    mo,
    np,
    pd,
    region_labels,
    segment,
):
    _bl1_df = pd.read_csv("result/baseline_1_client.csv")
    _bl2_df = pd.read_csv("result/baseline_2_clients.csv")
    _bl2_t = _bl2_df["time"].values.astype(float)
    _bl2_segs = segment(_bl2_t, _bl2_t, _bl2_t)
    _grid = np.arange(0, _bl2_t.max() + 60, 60)
    _bl2_power = np.zeros(len(_grid))
    for _seg_t, _, _ in _bl2_segs:
        _bl2_power[(_grid >= _seg_t[0]) & (_grid <= _seg_t[-1])] = POWER_TRAIN * 2

    exp_energy = np.sum(exp_total_power) * dt / 3600
    bl_energy = np.sum(bl_power) * dt / 3600
    bl2_energy = np.sum(_bl2_power) * 60 / 3600

    _EMA_ALPHA = 0.99
    _bl1_t = _bl1_df["time"].values.astype(float)
    _bl1_segs = segment(_bl1_t, _bl1_t, _bl1_t)
    _bl1_runtime = _bl1_t.max() / 3600
    _bl2_runtime = _bl2_t.max() / 3600
    _ours_ppl = pd.concat([
        pd.DataFrame({"time": _df["time"], "ppl": _df["perplexity"]}) for _df in client_dfs
    ]).groupby("time").mean().reset_index().sort_values("time")
    _exp_runtime = _ours_ppl["time"].max() / 3600

    _bl1_gpu_h = sum((_s[0][-1] - _s[0][0]) for _s in _bl1_segs) / 3600
    _bl2_gpu_h = sum((_s[0][-1] - _s[0][0]) for _s in _bl2_segs) / 3600 * 2
    _ours_gpu_h = 0
    for _df in client_dfs:
        _ct = _df["time"].values.astype(float)
        _csegs = segment(_ct, _ct, _ct)
        _ours_gpu_h += sum((_s[0][-1] - _s[0][0]) for _s in _csegs) / 3600

    _bl1_ppl_smooth = ema(_bl1_df["perplexity"].values, _EMA_ALPHA)
    _bl2_ppl_smooth = ema(_bl2_df["perplexity"].values, _EMA_ALPHA)
    _ours_ppl_smooth = ema(_ours_ppl["ppl"].values, _EMA_ALPHA)

    _table = "| Scenario | Runtime (h) | GPU Hours | Energy (kWh) | Best PPL (EMA) |\n|---|---|---|---|---|\n"
    _table += f"| Centralized | {_bl1_runtime:.1f} | {_bl1_gpu_h:.1f} | {bl_energy:.1f} | {_bl1_ppl_smooth.min():.1f} |\n"
    _table += f"| 2-Client FL | {_bl2_runtime:.1f} | {_bl2_gpu_h:.1f} | {bl2_energy:.1f} | {_bl2_ppl_smooth.min():.1f} |\n"
    _table += f"| Ours | {_exp_runtime:.1f} | {_ours_gpu_h:.1f} | {exp_energy:.1f} | {_ours_ppl_smooth.min():.1f} |\n"

    _carbon_table = "| Scenario | Carbon (kgCO\u2082) |\n|---|---|\n"
    for _r, _label in region_labels.items():
        _carbon_table += f"| Centralized {_label} | {bl_total_emissions[_r]:.2f} |\n"
    _carbon_table += f"| **Ours** | **{exp_total_emissions:.2f}** |\n"

    _pct_min = min(exp_total_emissions / bl_total_emissions[_r] * 100 for _r in region_labels)
    _pct_max = max(exp_total_emissions / bl_total_emissions[_r] * 100 for _r in region_labels)

    mo.md(f"""
    ### Training Summary

    {_table}

    *Best PPL is the minimum EMA-smoothed (α={_EMA_ALPHA}) perplexity reached during training. The EMA effective window is ~{int(1/(1-_EMA_ALPHA))} steps, so Best PPL approximates the best ~{int(1/(1-_EMA_ALPHA))}-batch moving average. GPU hours = sum of per-GPU training time across all clients.*

    ### Carbon Footprint

    {_carbon_table}

    Our approach emits only {_pct_min:.0f}\u2013{_pct_max:.0f}% of the carbon compared to single-region baselines.
    """)
    return


if __name__ == "__main__":
    app.run()
