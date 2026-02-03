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
    return mdates, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Training Loss Comparison

    Comparing the training loss of 1-client vs 2-client pilot over real time.
    """)
    return


@app.cell
def _(mdates, pd, plt):

    START_TIME = pd.Timestamp("2026-01-11 17:00:00+00:00")

    one_client_df = pd.read_csv("result/baseline_1_client.csv")
    two_client_df = pd.read_csv("result/baseline_2_clients.csv")

    fig, _ax = plt.subplots(figsize=(10, 6))

    def to_real_time(df):
        """Convert relative seconds to real timestamps"""
        return START_TIME + pd.to_timedelta(df["time"].astype(float), unit="s")

    def loss_col(df):
        """Get the training loss column names excluding MIN and MAX"""
        return df[next(c for c in df.columns if "train_loss" in c and "MIN" not in c and "MAX" not in c)]

    def calculate_smoothed_loss(series, beta=0.99):
        smoothed_values = []
        smooth_loss = 0
        for i, loss in enumerate(series):
            smooth_loss = beta * smooth_loss + (1 - beta) * loss
            debiased_loss = smooth_loss / (1 - beta**(i + 1))
            smoothed_values.append(debiased_loss)
        return smoothed_values

    _ax.plot(
        to_real_time(one_client_df),
        calculate_smoothed_loss(loss_col(one_client_df)),
        label="1 Client (Smoothed)",
        linestyle='-',
        alpha=0.7
    )

    _ax.plot(
        to_real_time(two_client_df),
        calculate_smoothed_loss(loss_col(two_client_df)),
        label="2 Clients (Smoothed)",
        linestyle='-',
        alpha=0.7
    )

    _ax.set_xlabel("Time")
    _ax.set_ylabel("Training Loss")
    _ax.set_ylim(2.5, 5.0)
    _ax.set_title("Training Loss Comparison")
    _ax.legend()
    _ax.grid(True, linestyle='--', alpha=0.6)
    _ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    fig
    return


@app.cell
def _(mo):
    mo.md("""
    # 3-Client Training Loss

    Individual loss curves for each client with discontinuous segments (gaps > 60s break the line).
    EMA smoothing (alpha=0.9) within continuous segments, original loss shown behind.
    """)
    return


@app.cell
def _(np, pd, plt):
    def _loss_col(df):
        """Get the training loss column (excluding MIN/MAX)"""
        return next(c for c in df.columns if "train_loss" in c and "MIN" not in c and "MAX" not in c)

    def _segment_and_smooth(times, losses, gap_threshold=60, alpha=0.9):
        """Segment data at gaps > threshold, compute EMA within each segment"""
        segments = []
        seg_times, seg_raw, seg_smooth = [], [], []
        ema = None

        for i, (t, loss) in enumerate(zip(times, losses)):
            if i > 0 and (t - times[i-1]) > gap_threshold:
                if seg_times:
                    segments.append((np.array(seg_times), np.array(seg_raw), np.array(seg_smooth)))
                seg_times, seg_raw, seg_smooth = [], [], []
                ema = None

            seg_times.append(t)
            seg_raw.append(loss)
            ema = loss if ema is None else alpha * ema + (1 - alpha) * loss
            seg_smooth.append(ema)

        if seg_times:
            segments.append((np.array(seg_times), np.array(seg_raw), np.array(seg_smooth)))
        return segments

    client_dfs = [pd.read_csv(f"result/client_{i}.csv") for i in range(3)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    _fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for idx, (df, ax, color) in enumerate(zip(client_dfs, axes, colors)):
        times = df["time"].values.astype(float)
        losses = df[_loss_col(df)].values
        segments = _segment_and_smooth(times, losses)

        for seg_t, seg_raw, seg_smooth in segments:
            ax.plot(seg_t / 3600, seg_raw, color=color, alpha=0.2, linewidth=1)
            ax.plot(seg_t / 3600, seg_smooth, color=color, alpha=1.0, linewidth=1.5)

        ax.set_ylabel(f"Client {idx}")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(2.0, 5.0)

    axes[-1].set_xlabel("Time (hours)")
    axes[0].set_title("Training Loss per Client (EMA α=0.9)")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mdates, pd, plt):


    mci = pd.read_csv("mci.csv", parse_dates=["point_time"], index_col="point_time")

    threshold = 100  # gCO2/kWh

    regions = ["SPP_TX", "CAISO_NORTH", "NEM_SA"]
    region_labels = {"SPP_TX": "Texas (SPP)", "CAISO_NORTH": "California (CAISO)", "NEM_SA": "South Australia (NEM)"}
    _colors = {"SPP_TX": "#d62728", "CAISO_NORTH": "#1f77b4", "NEM_SA": "#2ca02c"}

    _fig, _ax = plt.subplots(1, 1, figsize=(6, 3), sharex=True)

    for region in regions:
        series = mci[region].dropna()
        _ax.step(series.index, series.values, color=_colors[region], linewidth=1, label=region_labels[region])

        _ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)


    _ax.set(
        xlabel="",
        ylabel="gCO₂/kWh",
    )
    _ax.legend(loc='upper right')
    _ax.grid(True, linestyle='--', alpha=0.4)

    _ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    _fig.autofmt_xdate()

    plt.tight_layout()

    _fig
    return


if __name__ == "__main__":
    app.run()
