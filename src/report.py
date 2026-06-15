"""
Generate the risk report: console output + matplotlib charts saved to output/charts/.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.config import CHARTS_DIR, HOLDING_PERIOD_DAYS


def _fmt_usd(v: float) -> str:
    sign = "-" if v < 0 else "+"
    return f"{sign}${abs(v):,.0f}" if v != 0 else f"${v:,.0f}"


def _fmt_pos(v: float) -> str:
    return f"${v:,.0f}"


def print_inventory(summary_df: pd.DataFrame, total_mtm: float, total_pnl: float) -> None:
    print("\n" + "=" * 80)
    print("  SCRAPYARD RISK MODEL")
    print(f"  Report Date : {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  Horizon     : {HOLDING_PERIOD_DAYS} trading days")
    print("=" * 80)

    print("\n── INVENTORY (Mark-to-Market) ─────────────────────────────────────────────────────")
    hdr = (f"  {'Metal':<12} {'Qty (t)':>8} {'Avg Cost':>10} {'Spot':>10}"
           f" {'Break-even':>11} {'Basis':>7} {'MTM Value':>12} {'Unr. P&L':>12}"
           f" {'$/t Sens.':>10} {'Avg Days':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in summary_df.iterrows():
        print(
            f"  {r['metal'].capitalize():<12}"
            f" {r['total_tonnes']:>8.1f}"
            f" {_fmt_pos(r['avg_cost_per_tonne']):>10}"
            f" {_fmt_pos(r['spot_price']):>10}"
            f" {_fmt_pos(r['breakeven_price']):>11}"
            f" {r['avg_basis']:>7.1%}"
            f" {_fmt_pos(r['mtm_value']):>12}"
            f" {_fmt_usd(r['unrealised_pnl']):>12}"
            f" {r['sensitivity_per_dollar']:>10.1f}"
            f" {r['avg_days_held']:>9.0f}"
        )
    print(f"  {'TOTAL':<12} {'':>8} {'':>10} {'':>10} {'':>11} {'':>7}"
          f" {_fmt_pos(total_mtm):>12} {_fmt_usd(total_pnl):>12}"
          f" {summary_df['sensitivity_per_dollar'].sum():>10.1f}"
          f" {'':>9}")


def print_var(var_df: pd.DataFrame) -> None:
    print("\n── VALUE AT RISK ───────────────────────────────────────────────────")
    print(f"  {'Confidence':<12} {'Method':<15} {'VaR':>12} {'CVaR (ES)':>12} {'Divers. Benefit':>18}")
    print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*12} {'-'*18}")
    for _, r in var_df.iterrows():
        print(
            f"  {r['confidence']:<12}"
            f" {r['method']:<15}"
            f" {_fmt_pos(r['portfolio_var']):>12}"
            f" {_fmt_pos(r['portfolio_cvar']):>12}"
            f" {_fmt_pos(r['diversification_benefit']):>18}"
        )


def print_mc_horizons(mc_results: list[dict], horizons: list[int]) -> None:
    print("\n── MONTE CARLO — MULTI-HORIZON SUMMARY ────────────────────────────────")
    print(f"  {'Horizon':<10} {'Median':>14} {'5th Pctile':>14} {'95th Pctile':>14} {'P(Loss)':>9} {'MC VaR 95%':>12} {'MC CVaR 95%':>13}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*9} {'-'*12} {'-'*13}")
    for h, mc in zip(horizons, mc_results):
        m = mc["metrics"]
        print(
            f"  {str(h)+' days':<10}"
            f" {_fmt_pos(mc['percentiles'][50]):>14}"
            f" {_fmt_pos(mc['percentiles'][5]):>14}"
            f" {_fmt_pos(mc['percentiles'][95]):>14}"
            f" {m['prob_loss']:>9.1%}"
            f" {_fmt_pos(m['var_95']):>12}"
            f" {_fmt_pos(m['cvar_95']):>13}"
        )


def print_monte_carlo(mc_result: dict) -> None:
    print("\n── MONTE CARLO SIMULATION ──────────────────────────────────────────")
    print(f"  Initial Portfolio Value : {_fmt_pos(mc_result['initial_value'])}")
    m = mc_result["metrics"]
    print(f"  Probability of Loss     : {m['prob_loss']:.1%}")
    print(f"  Expected Loss (if loss) : {_fmt_pos(m['expected_loss_given_loss'])}")
    print(f"  95% VaR (MC)            : {_fmt_pos(m['var_95'])}")
    print(f"  95% CVaR (MC)           : {_fmt_pos(m['cvar_95'])}")
    print()
    print(f"  {'Percentile':<14} {'Portfolio Value':>16} {'P&L vs. Today':>16}")
    print(f"  {'-'*14} {'-'*16} {'-'*16}")
    init = mc_result["initial_value"]
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = mc_result["percentiles"][pct]
        print(f"  {pct:>3}th pctile   {_fmt_pos(val):>16} {_fmt_usd(val - init):>16}")


def print_scenarios(scenario_df: pd.DataFrame, ws: dict) -> None:
    print("\n── STRESS SCENARIOS ────────────────────────────────────────────────")
    metal_cols = [c for c in scenario_df.columns if c.endswith("_pnl") and c != "total_pnl"]
    header = f"  {'Scenario':<28}"
    for col in metal_cols:
        header += f" {col.replace('_pnl', '').capitalize():>11}"
    header += f" {'Total P&L':>11} {'% Change':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in scenario_df.iterrows():
        line = f"  {r['scenario']:<28}"
        for col in metal_cols:
            line += f" {_fmt_usd(r[col]):>11}"
        line += f" {_fmt_usd(r['total_pnl']):>11} {r['total_pnl_pct']:>8.1f}%"
        print(line)
    print(f"\n  Worst case: {ws['worst_scenario']} ({ws['worst_pnl_pct']:.1f}%)")
    best_sign = "+" if ws['best_pnl_pct'] >= 0 else ""
    print(f"  Best case : {ws['best_scenario']} ({best_sign}{ws['best_pnl_pct']:.1f}%)")
    print()


# Charts

def _save(fig: plt.Figure, name: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_price_history(prices: dict[str, pd.Series]) -> Path:
    metals = list(prices.keys())
    n = len(metals)
    ncols = 2
    nrows = (n + 1) // 2

    colors = {"copper": "#b87333", "aluminium": "#a8a9ad", "steel": "#607080", "stainless": "#6a6a7a"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharex=False)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, (metal, series) in zip(axes_flat, prices.items()):
        color = colors.get(metal, "#555")
        ax.plot(series.index, series.values, color=color, linewidth=1.2)
        ax.fill_between(series.index, series.values, series.values.min(), alpha=0.15, color=color)
        ax.set_title(f"{metal.capitalize()} — USD / Metric Tonne", fontsize=12)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Current price label on the right axis
        current = float(series.iloc[-1])
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        ax_r.set_yticks([current])
        ax_r.set_yticklabels([f"${current:,.0f}"], fontsize=8.5, color=color, fontweight="bold")
        ax_r.tick_params(axis="y", length=4, width=0.8, color=color)
        ax_r.spines[["top", "left", "bottom"]].set_visible(False)
        ax_r.spines["right"].set_color(color)
        ax_r.spines["right"].set_linewidth(0.8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return _save(fig, "price_history")


def chart_return_distributions(returns: dict[str, pd.Series]) -> Path:
    from scipy.stats import norm

    metals = list(returns.keys())
    n = len(metals)
    ncols = 2
    nrows = (n + 1) // 2

    colors = {
        "copper":    "#b87333",
        "aluminium": "#a8a9ad",
        "steel":     "#607080",
        "stainless": "#6a9ab0",
        "brass":     "#c9a84c",
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, metal in zip(axes_flat, metals):
        r = returns[metal]
        color = colors.get(metal, "#555")
        ax.hist(r.values, bins=60, density=True, color=color, alpha=0.6, label="Empirical")
        x = np.linspace(r.min(), r.max(), 200)
        ax.plot(x, norm.pdf(x, r.mean(), r.std()), "k--", linewidth=1.2, label="Normal fit")
        q5 = np.percentile(r.values, 5)
        ax.axvline(q5, color="red", linewidth=1, linestyle=":", label=f"5th pctile ({q5:.2%})")
        ax.set_title(f"{metal.capitalize()} Daily Returns", fontsize=11)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return _save(fig, "return_distributions")


def _draw_mc_panel(ax: plt.Axes, mc_result: dict) -> None:
    from matplotlib.transforms import blended_transform_factory, offset_copy

    paths = mc_result["paths"]
    init = mc_result["initial_value"]
    n_steps = paths.shape[1]
    horizon = n_steps - 1
    x = np.arange(n_steps)

    sample_idx = np.random.choice(len(paths), size=min(300, len(paths)), replace=False)
    for i in sample_idx:
        ax.plot(x, paths[i], color="#1f77b4", alpha=0.04, linewidth=0.5)

    pcts = {p: np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}
    ax.fill_between(x, pcts[5], pcts[95], alpha=0.15, color="#1f77b4", label="5th–95th pctile")
    ax.fill_between(x, pcts[25], pcts[75], alpha=0.25, color="#1f77b4", label="25th–75th pctile")
    ax.plot(x, pcts[50], color="#1f77b4", linewidth=2, label="Median")
    ax.axhline(init, color="black", linewidth=1, linestyle="--", label="Today")

    m = mc_result["metrics"]
    subtitle = (
        f"5th pctile: ${mc_result['percentiles'][5]:,.0f}  |  "
        f"Median: ${mc_result['percentiles'][50]:,.0f}  |  "
        f"95th pctile: ${mc_result['percentiles'][95]:,.0f}  |  "
        f"P(loss): {m['prob_loss']:.1%}"
    )
    ax.set_title(f"{horizon}-Day Horizon — {subtitle}", fontsize=10)
    ax.set_xlabel("Trading Days", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    nudge_needed = abs(pcts[50][-1] - init) < y_range * 0.03
    median_label_pos = init + y_range * 0.03 if nudge_needed else pcts[50][-1]

    if nudge_needed:
        tick_positions = [pcts[5][-1], pcts[25][-1], init,      pcts[75][-1], pcts[95][-1]]
        tick_labels    = [pcts[5][-1], pcts[25][-1], init,      pcts[75][-1], pcts[95][-1]]
        tick_colors    = ["#1f77b4",   "#3a86c8",    "#333333", "#3a86c8",    "#1f77b4"   ]

        fig = ax.get_figure()
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        ax_r.spines[["top", "left"]].set_visible(False)

        trans_spine    = blended_transform_factory(ax_r.transAxes, ax_r.transData)
        trans_tick_end = blended_transform_factory(
            offset_copy(ax_r.transAxes, fig=fig, x=4, units="points"), ax_r.transData
        )
        trans_label    = blended_transform_factory(
            offset_copy(ax_r.transAxes, fig=fig, x=9, units="points"), ax_r.transData
        )
        ax_r.annotate(
            "",
            xy=(1.0, median_label_pos),  xycoords=trans_tick_end,
            xytext=(1.0, pcts[50][-1]),  textcoords=trans_spine,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8, shrinkA=0, shrinkB=0),
            clip_on=False,
        )
        ax_r.text(
            1.0, median_label_pos, f"${pcts[50][-1]:,.0f}",
            transform=trans_label, fontsize=7.5, color="#1f77b4",
            va="center", ha="left", clip_on=False,
        )
    else:
        tick_positions = [pcts[5][-1], pcts[25][-1], init,      pcts[50][-1], pcts[75][-1], pcts[95][-1]]
        tick_labels    = tick_positions[:]
        tick_colors    = ["#1f77b4",   "#3a86c8",    "#333333", "#1f77b4",    "#3a86c8",    "#1f77b4"   ]
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        ax_r.spines[["top", "left"]].set_visible(False)

    ax_r.set_yticks(tick_positions)
    ax_r.set_yticklabels([f"${v:,.0f}" for v in tick_labels], fontsize=7.5)
    for tick_label, color in zip(ax_r.get_yticklabels(), tick_colors):
        tick_label.set_color(color)
    ax_r.tick_params(axis="y", length=4, width=0.8)


def chart_monte_carlo(mc_results: list[dict], label: str = "") -> Path:
    n = len(mc_results)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    n_sims = len(mc_results[0]["paths"])
    prefix = f"{label.capitalize()} — " if label else ""
    fig.suptitle(
        f"{prefix}Monte Carlo Portfolio Value ({n_sims:,} simulations)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for ax, mc in zip(axes_flat, mc_results):
        _draw_mc_panel(ax, mc)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout(w_pad=4, h_pad=3)
    name = f"monte_carlo_{label}" if label else "monte_carlo"
    return _save(fig, name)


def chart_scenarios(scenario_df: pd.DataFrame, label: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in scenario_df["total_pnl"]]
    bars = ax.barh(scenario_df["scenario"], scenario_df["total_pnl"], color=colors, alpha=0.8)
    ax.bar_label(bars, fmt=lambda v: f"${v:,.0f}", padding=5, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    prefix = f"{label.capitalize()} — " if label else ""
    ax.set_title(f"{prefix}Stress Scenario P&L Impact", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 1.18, xmax)
    fig.tight_layout()
    name = f"scenarios_{label}" if label else "scenarios"
    return _save(fig, name)


def chart_commodity_breakdown(summary_df: pd.DataFrame) -> Path:
    
    def _abbrev(v: float) -> str:
        if v >= 1_000_000:
            return f"${v / 1_000_000:.2f}M"
        return f"${v / 1_000:.0f}K"

    df = summary_df.copy()
    df["label"] = df["metal"].str.capitalize()
    df = df.sort_values("mtm_value")

    metal_colors = {
        "copper":    "#b87333",
        "aluminium": "#a8a9ad",
        "steel":     "#607080",
        "stainless": "#6a9ab0",
        "brass":     "#c9a84c",
        "lead":      "#475569",
        "zinc":      "#7c8f3a",
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    y = np.arange(len(df))

    book_colors = [metal_colors.get(m, "#999") for m in df["metal"]]
    gain_colors = [metal_colors.get(m, "#999") for m in df["metal"]]

    ax.barh(y, df["book_value"],                        color=book_colors, alpha=0.40, label="Book Cost")
    ax.barh(y, df["unrealised_pnl"].clip(lower=0),
            left=df["book_value"],                      color=gain_colors, alpha=0.85, label="Unrealised Gain")

    max_mtm = df["mtm_value"].max()
    x_max = max_mtm * 1.28
    min_book = df["book_value"].min()
    if min_book / x_max < 0.05:
        x_max = max(x_max, min_book / 0.05)
    ax.set_xlim(0, x_max)

    for i, (_, row) in enumerate(df.iterrows()):
        book = row["book_value"]
        mtm  = row["mtm_value"]

        ax.text(
            book / 2, i,
            _abbrev(book),
            va="center", ha="center", fontsize=8.5, fontweight="bold", color="#222",
        )

        ax.text(
            mtm + x_max * 0.008, i,
            f"${mtm:,.0f}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=10)
    ax.set_title("Portfolio MTM Value by Commodity", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K"))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "commodity_breakdown")


def chart_var_summary(var_df: pd.DataFrame, label: str = "") -> Path:
    pivot = var_df[var_df["confidence"] == "95%"].set_index("method")[["portfolio_var", "portfolio_cvar"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot))
    width = 0.35

    bars_var  = ax.bar(x - width / 2, pivot["portfolio_var"],  width, label="VaR",      color="#1f77b4", alpha=0.8)
    bars_cvar = ax.bar(x + width / 2, pivot["portfolio_cvar"], width, label="CVaR (ES)", color="#ff7f0e", alpha=0.8)

    for bar in bars_var:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 0.94, f"${h:,.0f}",
                ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    for bar in bars_cvar:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 0.94, f"${h:,.0f}",
                ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    prefix = f"{label.capitalize()} — " if label else ""
    ax.set_title(f"{prefix}95% VaR vs CVaR by Method", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    name = f"var_summary_{label}" if label else "var_summary"
    return _save(fig, name)


def chart_aging(summary_df: pd.DataFrame) -> Path:

    def _aging_color(days: float) -> str:
        if days <= 30:
            return "#059669"
        if days <= 60:
            return "#d97706"
        return "#dc2626"

    df = summary_df.sort_values("avg_days_held")
    labels = [m.capitalize() for m in df["metal"]]
    avg_days = df["avg_days_held"].values
    max_days = df["max_days_held"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    colors_avg = [_aging_color(d) for d in avg_days]
    bars1 = ax1.barh(labels, avg_days, color=colors_avg, alpha=0.85)
    ax1.bar_label(bars1, fmt=lambda v: f"{v:.0f} days", padding=5, fontsize=9)
    ax1.axvline(30, color="#d97706", linewidth=1, linestyle="--", alpha=0.6, label="30-day threshold")
    ax1.axvline(60, color="#dc2626", linewidth=1, linestyle="--", alpha=0.6, label="60-day threshold")
    ax1.set_title("Average Days Held by Metal", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Calendar Days", fontsize=9)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    max_x = max(avg_days.max() * 1.25, 70)
    ax1.set_xlim(0, max_x)

    colors_max = [_aging_color(d) for d in max_days]
    bars2 = ax2.barh(labels, max_days, color=colors_max, alpha=0.85)
    ax2.bar_label(bars2, fmt=lambda v: f"{v:.0f} days", padding=5, fontsize=9)
    ax2.axvline(30, color="#d97706", linewidth=1, linestyle="--", alpha=0.6)
    ax2.axvline(60, color="#dc2626", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_title("Oldest Lot Held by Metal", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Calendar Days", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    max_x2 = max(max_days.max() * 1.25, 70)
    ax2.set_xlim(0, max_x2)

    fig.suptitle("Inventory Aging Report — Days Held per Metal", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "inventory_aging")
