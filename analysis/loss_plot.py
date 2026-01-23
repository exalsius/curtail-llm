import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Training Loss Comparison

    Comparing the training loss of Centralized Baseline vs Pilot Baseline over time.
    """)
    return


@app.cell
def _(pd):
    # Load data
    # Using 'centalized' (typo in filename respected)
    centralized_df = pd.read_csv("centalized_baseline.csv")
    pilot_df = pd.read_csv("pilot_baseline.csv")
    return centralized_df, pilot_df


@app.cell
def _(centralized_df, pilot_df):
    # Inspect columns to identify the correct loss columns
    print("Centralized columns:", centralized_df.columns.tolist())
    print("Pilot columns:", pilot_df.columns.tolist())

    # Identify Pilot loss column (contains 'train_loss' and not MIN/MAX)
    pilot_loss_col = [c for c in pilot_df.columns if "train_loss" in c and "MIN" not in c and "MAX" not in c][0]

    # Identify Centralized loss column
    centralized_loss_col = "baseline - train/loss"

    # Define smoothing function (same as base_train.py)
    def calculate_smoothed_loss(series, beta=0.9):
        smoothed_values = []
        smooth_loss = 0
        for i, loss in enumerate(series):
            # If it's the first step, initialize with the first value (common practice)
            # base_train.py initializes smooth_loss=0, but let's stick to its formula exactly:
            # smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
            smooth_loss = beta * smooth_loss + (1 - beta) * loss
            debiased_loss = smooth_loss / (1 - beta**(i + 1))
            smoothed_values.append(debiased_loss)
        return smoothed_values

    # Apply smoothing to Pilot loss
    pilot_df["smoothed_loss"] = calculate_smoothed_loss(pilot_df[pilot_loss_col])
    return (centralized_loss_col,)


@app.cell
def _(centralized_df, centralized_loss_col, pilot_df, plt):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Centralized Baseline
    ax.plot(
        centralized_df["total_training_time"], 
        centralized_df[centralized_loss_col], 
        label="Centralized Baseline",
        linestyle='-',
        alpha=0.7
    )

    # Plot Pilot Baseline (Smoothed)
    ax.plot(
        pilot_df["time"], 
        pilot_df["smoothed_loss"], 
        label="Pilot Baseline (Smoothed)",
        linestyle='-',
        alpha=0.7
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Training Loss")
    ax.set_ylim(2.6, 3.8)
    ax.set_title("Final Loss Comparison (Time Domain)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
