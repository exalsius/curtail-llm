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

    def draw_bar(ax, labels, values, colors, ylabel, fmt="{:.1f}", width=0.7, ylim_factor=1.15, offset=0.5, ann_fontsize=9, horizontal=False):
        if horizontal:
            bars = ax.barh(labels, values, color=colors, height=width)
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                        fmt.format(val), ha="left", va="center", fontsize=ann_fontsize)
            ax.set_xlabel(ylabel, fontsize=FONTSIZE)
            ax.tick_params(labelsize=FONTSIZE, axis="y")
            ax.tick_params(labelsize=FONTSIZE, axis="x")
            ax.grid(True, linestyle="--", alpha=0.4, axis="x")
            ax.set_xlim(0, max(values) * ylim_factor)
        else:
            bars = ax.bar(labels, values, color=colors, width=width)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                        fmt.format(val), ha="center", va="bottom", fontsize=ann_fontsize)
            ax.set_ylabel(ylabel, fontsize=FONTSIZE)
            ax.tick_params(labelsize=8, labelrotation=30, axis="x")
            ax.tick_params(labelsize=FONTSIZE, axis="y")
            ax.grid(True, linestyle="--", alpha=0.4, axis="y")
            ax.set_ylim(0, max(values) * ylim_factor)
            plt.setp(ax.get_xticklabels(), ha="right")

    client_dfs = [pd.read_csv(f"evaluation/results/exp1/client_{i}.csv") for i in range(3)]
    bl1_df = pd.read_csv("evaluation/results/baseline_1_client.csv")
    bl2_df = pd.read_csv("evaluation/results/baseline_2_clients.csv")
    events_df = pd.read_csv("evaluation/results/exp1/events.csv")
    mci = pd.read_csv("evaluation/mci.csv", parse_dates=["point_time"], index_col="point_time")

    ours_ppl = pd.concat([
        pd.DataFrame({"time": _df["time"], "ppl": _df["perplexity"]}) for _df in client_dfs
    ]).groupby("time").mean().reset_index().sort_values("time")

    region_labels = {"CAISO_NORTH": "California", "SPP_TX": "Texas", "NEM_SA": "South Australia", "DE": "Germany"}
    region_colors = {"CAISO_NORTH": "#1f77b4", "SPP_TX": "#d62728", "NEM_SA": "#2ca02c", "DE": "#9467bd"}
    region_list = ["CAISO_NORTH", "SPP_TX", "NEM_SA", "DE"]
    return (
        COLORS,
        FONTSIZE,
        START_TIME,
        bl1_df,
        bl2_df,
        client_dfs,
        draw_bar,
        ema,
        events_df,
        mci,
        mo,
        np,
        ours_ppl,
        plt,
        region_colors,
        region_labels,
        region_list,
        segment,
        style_time_axes,
    )


@app.cell
def _(
    FONTSIZE,
    START_TIME,
    client_dfs,
    ema,
    mci,
    plt,
    region_colors,
    region_labels,
    region_list,
    segment,
    style_time_axes,
):
    _fig, axes = plt.subplots(4, 1, figsize=(7, 5), sharex=True, gridspec_kw=dict(height_ratios=[1.2,1,1,1]))

    mci_hours = (mci.index - START_TIME).total_seconds() / 3600
    regions = {r: (region_labels[r], region_colors[r]) for r in region_list}

    for _region, (_label, _color) in regions.items():
        _mask = mci[_region].notna()
        axes[0].step(mci_hours[_mask], mci[_region][_mask].values, color=_color, linewidth=1.2, label=_label)

    axes[0].axhline(y=100, color='#333', linestyle='--', linewidth=1, alpha=0.9, label="Curtailment threshold")
    axes[0].set_ylabel("Carbon intensity\n(gCO₂/kWh)", fontsize=FONTSIZE)
    axes[0].legend(loc='upper right', fontsize=8, framealpha=1)
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

        _ax.set_ylabel("Train loss", fontsize=FONTSIZE)
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
        title="training stops if no sites\nare curtailed",
        indicator_ax=axes[1],
    )
    _add_inset(
        pos=[0.43, 0.40, 0.20, 0.10], client=0,
        xlim=(15500/3600, 18000/3600), ylim=(2.8, 3.4),
        title="if more than one site is curtailed,\ntraining switches to federated mode",
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
    return


@app.cell
def _(COLORS, FONTSIZE, bl1_df, bl2_df, ema, ours_ppl, plt, style_time_axes):
    _fig, _ax = plt.subplots(figsize=(4, 2.5))

    def _plot(ax, hours, raw, color, label):
        smoothed = ema(raw, 0.99)
        ax.plot(hours, raw, color=color, alpha=0.1, linewidth=1)
        ax.plot(hours, smoothed, color=color, linewidth=1, label=label)
        ax.plot(hours[-1], smoothed[-1], 'o', color=color, markersize=3)

    _plot(_ax, (bl1_df["time"] / 3600).values, bl1_df["perplexity"].values, COLORS["bl1"], "Centralized")
    _plot(_ax, (bl2_df["time"] / 3600).values, bl2_df["perplexity"].values, COLORS["bl2"], "Two-site FL")
    _plot(_ax, (ours_ppl["time"] / 3600).values, ours_ppl["ppl"].values, COLORS["ours"], "Ours")

    _ax.set_ylabel("Train perplexity", fontsize=FONTSIZE)
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
    bl1_df,
    client_dfs,
    events_df,
    mci,
    np,
    plt,
    region_colors,
    region_labels,
    region_list,
    segment,
    style_time_axes,
):
    POWER_IDLE = 0.468
    POWER_TRAIN = 2.024

    client_regions = {"client_0": "CAISO_NORTH", "client_1": "SPP_TX", "client_2": "NEM_SA"}

    prov_windows = {f"client_{i}": [] for i in range(3)}
    prov_start = {}
    for _, row in events_df.iterrows():
        evt, client = row["event_type"], row["client"]
        if evt == "PROVISION_COMPLETE":
            prov_start[client] = row["elapsed_s"]
        elif evt == "DEPROVISION" and client in prov_start:
            prov_windows[client].append((prov_start.pop(client), row["elapsed_s"]))
    exp_end = events_df["elapsed_s"].max()
    for client, start in prov_start.items():
        prov_windows[client].append((start, exp_end))

    train_segments = {}
    for _i, _df in enumerate(client_dfs):
        _t = _df["time"].values.astype(float)
        _segs = segment(_t, _t, _t)
        train_segments[f"client_{_i}"] = [(_s[0][0], _s[0][-1]) for _s in _segs]

    max_time = max(bl1_df["time"].astype(float).max(), exp_end)
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

    bl_mask = (grid >= bl1_df["time"].astype(float).min()) & (grid <= bl1_df["time"].astype(float).max())
    bl_power = np.where(bl_mask, POWER_TRAIN, 0.0)

    mci_elapsed = (mci.index - START_TIME).total_seconds().values
    mci_on_grid = {}
    for r in region_list:
        mci_on_grid[r] = np.interp(grid, mci_elapsed, mci[r].values)

    exp_emission_rate = sum(exp_powers[c] * mci_on_grid[r] for c, r in client_regions.items())

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
        bl_power,
        bl_total_emissions,
        client_regions,
        dt,
        exp_powers,
        exp_total_emissions,
        exp_total_power,
        grid,
        mci_on_grid,
    )


@app.cell
def _(
    COLORS,
    bl_total_emissions,
    draw_bar,
    exp_total_emissions,
    plt,
    region_colors,
    region_labels,
    region_list,
):
    _fig, _ax_bar = plt.subplots(1, 1, figsize=(4, 2.5))

    _bar_labels = [region_labels[_r].replace(" ", "\n") for _r in region_list] + ["Ours\n(distributed)"]
    _bar_values = [bl_total_emissions[_r] for _r in region_list] + [exp_total_emissions]
    _bar_colors = [region_colors[_r] for _r in region_list] + [COLORS["ours"]]
    draw_bar(_ax_bar, _bar_labels, _bar_values, _bar_colors, "Total emissions (kgCO\u2082)", ylim_factor=1.1, offset=0.3)

    _fig.savefig("figures/emissions_short.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    COLORS,
    POWER_TRAIN,
    bl1_df,
    bl_total_emissions,
    client_regions,
    dt,
    exp_powers,
    exp_total_emissions,
    grid,
    mci_on_grid,
    np,
    region_colors,
    region_labels,
    region_list,
    segment,
):
    CURTAIL_THRESHOLD = 100  # gCO₂/kWh
    curtail_masks = {r: mci_on_grid[r] < CURTAIL_THRESHOLD for r in region_list}

    def _segs_to_grid(times):
        _s = segment(times, times, times)
        _mask = np.zeros(len(grid), dtype=bool)
        for _st, _, _ in _s:
            _mask[(grid >= _st[0]) & (grid <= _st[-1])] = True
        return _mask

    _bl1_power_arr = np.where(_segs_to_grid(bl1_df["time"].values.astype(float)), POWER_TRAIN, 0.0)

    curt_fracs = []
    _den = np.sum(_bl1_power_arr) * dt
    for _r in region_list:
        _num = np.sum(_bl1_power_arr * curtail_masks[_r]) * dt
        curt_fracs.append(_num / _den if _den > 0 else 0.0)

    _ours_num = sum(np.sum(exp_powers[c] * curtail_masks[r]) * dt for c, r in client_regions.items())
    _ours_den = sum(np.sum(exp_powers[c]) * dt for c in client_regions)
    curt_fracs.append(_ours_num / _ours_den if _ours_den > 0 else 0.0)

    bar_labels = [region_labels[r] for r in region_list] + ["Ours"]
    bar_colors = [region_colors[r] for r in region_list] + [COLORS["ours"]]
    em_values = [bl_total_emissions[r] for r in region_list] + [exp_total_emissions]
    return bar_colors, bar_labels, curt_fracs, em_values


@app.cell
def _(bar_colors, bar_labels, curt_fracs, draw_bar, plt):
    _fig, _ax = plt.subplots(figsize=(3.5, 1.8))
    draw_bar(_ax, bar_labels, [f * 100 for f in curt_fracs], bar_colors,
             "Curtailed energy (%)", fmt="{:.0f}%", ylim_factor=1.2, horizontal=True)
    _fig.savefig("figures/barchart_curtailed.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(bar_colors, bar_labels, draw_bar, em_values, plt):
    _fig, _ax = plt.subplots(figsize=(3.5, 1.8))
    draw_bar(_ax, bar_labels, em_values, bar_colors, "Total emissions (kgCO\u2082)", horizontal=True)
    _fig.savefig("figures/barchart_emissions.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    POWER_TRAIN,
    bl1_df,
    bl2_df,
    bl_power,
    bl_total_emissions,
    client_dfs,
    dt,
    ema,
    exp_total_emissions,
    exp_total_power,
    mo,
    np,
    ours_ppl,
    region_labels,
    segment,
):
    _bl2_t = bl2_df["time"].values.astype(float)
    _bl2_segs = segment(_bl2_t, _bl2_t, _bl2_t)
    _grid = np.arange(0, _bl2_t.max() + 60, 60)
    _bl2_power = np.zeros(len(_grid))
    for _seg_t, _, _ in _bl2_segs:
        _bl2_power[(_grid >= _seg_t[0]) & (_grid <= _seg_t[-1])] = POWER_TRAIN * 2

    exp_energy = np.sum(exp_total_power) * dt / 3600
    bl_energy = np.sum(bl_power) * dt / 3600
    bl2_energy = np.sum(_bl2_power) * 60 / 3600

    _EMA_ALPHA = 0.99
    _bl1_t = bl1_df["time"].values.astype(float)
    _bl1_segs = segment(_bl1_t, _bl1_t, _bl1_t)
    _bl1_runtime = _bl1_t.max() / 3600
    _bl2_runtime = _bl2_t.max() / 3600
    _exp_runtime = ours_ppl["time"].max() / 3600

    _bl1_gpu_h = sum((_s[0][-1] - _s[0][0]) for _s in _bl1_segs) / 3600
    _bl2_gpu_h = sum((_s[0][-1] - _s[0][0]) for _s in _bl2_segs) / 3600 * 2
    _ours_gpu_h = 0
    for _df in client_dfs:
        _ct = _df["time"].values.astype(float)
        _csegs = segment(_ct, _ct, _ct)
        _ours_gpu_h += sum((_s[0][-1] - _s[0][0]) for _s in _csegs) / 3600

    _bl1_ppl_smooth = ema(bl1_df["perplexity"].values, _EMA_ALPHA)
    _bl2_ppl_smooth = ema(bl2_df["perplexity"].values, _EMA_ALPHA)
    _ours_ppl_smooth = ema(ours_ppl["ppl"].values, _EMA_ALPHA)

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
