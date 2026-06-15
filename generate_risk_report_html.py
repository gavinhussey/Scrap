"""Generate risk_report.html from the current risk model outputs."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.inventory import (
    exposure_by_metal,
    load_inventory,
    mark_to_market,
    portfolio_summary,
    total_exposure,
)
from src.monte_carlo import simulate
from src.prices import daily_returns, fetch_all_prices, merge_exposures_by_driver
from src.scenarios import run_scenarios, worst_case_summary
from src.var_model import var_table


OUT = Path("risk_report.html")


COLORS = {
    "aluminium": "#94a3b8",
    "brass": "#c9a84c",
    "copper": "#b87333",
    "lead": "#475569",
    "stainless": "#6a9ab0",
    "steel": "#607080",
    "zinc": "#7c8f3a",
}


def usd(v: float, signed: bool = False, compact: bool = False) -> str:
    sign = ""
    if signed:
        sign = "+" if v >= 0 else "-"
    n = abs(v) if signed else v
    if compact and abs(n) >= 1_000:
        return f"{sign}${n / 1_000:,.0f}K"
    return f"{sign}${n:,.0f}"


def pct(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}%"


def cls(v: float) -> str:
    return "pos" if v >= 0 else "neg"


def esc(v: object) -> str:
    return html.escape(str(v))


def table(rows: str, head: str | None = None, foot: str | None = None) -> str:
    h = f"<thead>{head}</thead>" if head else ""
    f = f"<tfoot>{foot}</tfoot>" if foot else ""
    return f"<table>{h}<tbody>{rows}</tbody>{f}</table>"


def chart(path: str, label: str) -> str:
    return f"""
    <div class="chart-card">
      <div class="chart-label">{esc(label)}</div>
      <img src="{esc(path)}" alt="{esc(label)}" />
    </div>
    """


def build_html(refresh: bool = False) -> str:
    prices = fetch_all_prices(force_refresh=refresh)
    spot_prices = {m: float(s.iloc[-1]) for m, s in prices.items()}

    inventory = load_inventory()
    mtm = mark_to_market(inventory, spot_prices)
    summary = portfolio_summary(mtm).sort_values("mtm_value", ascending=False)
    total_mtm = total_exposure(mtm)
    total_book = float(mtm["book_value"].sum())
    total_pnl = float(mtm["unrealised_pnl"].sum())
    total_tonnes = float(summary["total_tonnes"].sum())
    exposures = exposure_by_metal(mtm)

    returns_map = {m: daily_returns(prices[m]) for m in exposures}
    mc_exposures, mc_returns = merge_exposures_by_driver(exposures, returns_map)
    var_df = var_table(mc_exposures, mc_returns)
    horizons = [30, 60, 90, 180]
    mc_results = [simulate(mc_exposures, prices, horizon=h, seed=42) for h in horizons]
    mc = mc_results[0]
    scenario_df = run_scenarios(exposures)
    ws = worst_case_summary(scenario_df)

    report_dt = datetime.today()
    report_date_long = report_dt.strftime("%B %-d, %Y")
    report_date_iso = report_dt.strftime("%Y-%m-%d")

    var95 = var_df[(var_df["confidence"] == "95%") & (var_df["method"] == "Parametric")].iloc[0]
    pnl_pct = total_pnl / total_book * 100 if total_book else 0

    inv_rows = ""
    for _, r in summary.iterrows():
        metal = str(r["metal"])
        inv_rows += f"""
        <tr>
          <td><span class="dot" style="background:{COLORS.get(metal, '#64748b')}"></span>{esc(metal.capitalize())}</td>
          <td>{r['total_tonnes']:,.1f}</td>
          <td>{usd(r['avg_cost_per_tonne'])}</td>
          <td>{usd(r['spot_price'])}</td>
          <td>{usd(r['breakeven_price'])}</td>
          <td>{r['avg_basis']:.1%}</td>
          <td>{usd(r['book_value'])}</td>
          <td>{usd(r['mtm_value'])}</td>
          <td class="{cls(r['unrealised_pnl'])}">{usd(r['unrealised_pnl'], signed=True)}</td>
          <td>{r['sensitivity_per_dollar']:,.1f}</td>
          <td>{r['avg_days_held']:,.0f}</td>
        </tr>
        """
    inv_foot = f"""
      <tr>
        <td>Total</td><td>{total_tonnes:,.1f}</td><td></td><td></td><td></td><td></td>
        <td>{usd(total_book)}</td><td>{usd(total_mtm)}</td>
        <td class="{cls(total_pnl)}">{usd(total_pnl, signed=True)}</td>
        <td>{summary['sensitivity_per_dollar'].sum():,.1f}</td><td></td>
      </tr>
    """

    var_rows = ""
    for _, r in var_df.iterrows():
        var_rows += f"""
        <tr>
          <td>{esc(r['confidence'])}</td><td>{esc(r['method'])}</td>
          <td>{usd(r['portfolio_var'])}</td><td>{usd(r['portfolio_cvar'])}</td>
          <td>{usd(r['diversification_benefit'])}</td>
        </tr>
        """

    mc_ladder = ""
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = mc["percentiles"][p]
        diff = val - mc["initial_value"]
        mc_ladder += f"<tr><td>{p}th</td><td>{usd(val)}</td><td class=\"{cls(diff)}\">{usd(diff, signed=True)}</td></tr>"

    horizon_rows = ""
    for h, res in zip(horizons, mc_results):
        m = res["metrics"]
        horizon_rows += f"""
        <tr>
          <td>{h} days</td><td>{usd(res['percentiles'][50])}</td>
          <td class="neg">{usd(res['percentiles'][5])}</td>
          <td class="pos">{usd(res['percentiles'][95])}</td>
          <td>{m['prob_loss']:.1%}</td><td>{usd(m['var_95'])}</td><td>{usd(m['cvar_95'])}</td>
        </tr>
        """

    scen_cols = [c for c in scenario_df.columns if c.endswith("_pnl") and c != "total_pnl"]
    scen_head = "<tr><th>Scenario</th>" + "".join(f"<th>{esc(c[:-4].capitalize())}</th>" for c in scen_cols) + "<th>Total P&L</th><th>% Change</th></tr>"
    scen_rows = ""
    for _, r in scenario_df.iterrows():
        cells = "".join(f"<td class=\"{cls(r[c])}\">{usd(r[c], signed=True, compact=True)}</td>" for c in scen_cols)
        scen_rows += f"<tr><td>{esc(r['scenario'])}</td>{cells}<td class=\"{cls(r['total_pnl'])}\">{usd(r['total_pnl'], signed=True, compact=True)}</td><td class=\"{cls(r['total_pnl_pct'])}\">{pct(r['total_pnl_pct'])}</td></tr>"

    commodity_sections = ""
    for _, r in summary.iterrows():
        metal = str(r["metal"])
        commodity_sections += f"""
        <section class="commodity">
          <div class="commodity-title">
            <span class="swatch" style="background:{COLORS.get(metal, '#64748b')}"></span>
            <div>
              <h3>{esc(metal.capitalize())}</h3>
              <p>Book {usd(r['book_value'])} · MTM {usd(r['mtm_value'])} · {r['total_tonnes']:,.1f} t · {r['mtm_value'] / total_mtm:.1%} of portfolio · Break-even {usd(r['breakeven_price'])}/t · Sensitivity {r['sensitivity_per_dollar']:,.1f}/t</p>
            </div>
          </div>
          <div class="grid three">
            {chart(f"output/charts/monte_carlo_{metal}.png", f"{metal.capitalize()} Monte Carlo")}
            {chart(f"output/charts/var_summary_{metal}.png", f"{metal.capitalize()} VaR")}
            {chart(f"output/charts/scenarios_{metal}.png", f"{metal.capitalize()} Stress")}
          </div>
        </section>
        """

    sources = ""
    for metal, cfg in config.METALS.items():
        ticker = cfg.get("ticker", f"Proxy: {cfg.get('price_proxy')}")
        sources += f"<tr><td>{esc(metal.capitalize())}</td><td><code>{esc(ticker)}</code></td><td>{esc('Yahoo Finance' if 'ticker' in cfg else 'Proxy series')}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BMR / MMR Scrapyard Risk Report</title>
  <style>
    :root {{ --bg:#f4f6f8; --card:#fff; --text:#172033; --muted:#667085; --border:#d9e0e8; --header:#111827; --blue:#2563eb; --green:#047857; --red:#b91c1c; --amber:#b45309; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); font-size:14px; line-height:1.55; }}
    .container {{ max-width:1360px; margin:0 auto; padding:0 28px; }}
    header {{ background:var(--header); color:white; padding:32px 0; position:sticky; top:0; z-index:5; border-bottom:1px solid rgba(255,255,255,.12); }}
    .head {{ display:flex; justify-content:space-between; align-items:center; gap:24px; }}
    h1 {{ margin:0; font-size:24px; letter-spacing:0; }}
    .sub {{ margin-top:4px; color:#9ca3af; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .stats {{ display:flex; gap:28px; text-align:right; }}
    .label {{ color:#9ca3af; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
    .value {{ font-size:20px; font-weight:700; margin-top:2px; }}
    nav {{ background:white; border-bottom:1px solid var(--border); }}
    nav a {{ display:inline-block; color:var(--muted); text-decoration:none; padding:13px 16px; font-weight:600; font-size:13px; }}
    nav a:hover {{ color:var(--blue); }}
    main {{ padding:34px 0 72px; }}
    .notice {{ background:#ecfdf5; color:#065f46; border-bottom:1px solid #a7f3d0; padding:10px 0; font-size:12px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:16px; margin-bottom:32px; }}
    .card, .chart-card, .commodity {{ background:var(--card); border:1px solid var(--border); border-radius:8px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
    .card {{ padding:20px; overflow:auto; }}
    .kpi {{ padding:20px; }}
    .kpi .value {{ color:var(--text); font-size:28px; }}
    .delta {{ color:var(--muted); font-size:12px; margin-top:8px; }}
    section {{ margin-top:34px; }}
    .section-head {{ display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:14px; }}
    h2 {{ margin:0; font-size:21px; }}
    h3 {{ margin:0; font-size:17px; }}
    .desc {{ color:var(--muted); margin-top:3px; }}
    .badge {{ border:1px solid #bbf7d0; background:#f0fdf4; color:#166534; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:700; }}
    .grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:18px; }}
    .grid.three {{ grid-template-columns:repeat(3, minmax(0, 1fr)); }}
    .span2 {{ grid-column:1 / -1; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; white-space:nowrap; }}
    th {{ text-align:left; color:var(--muted); border-bottom:1px solid var(--border); padding:9px 10px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
    td {{ border-bottom:1px solid #eef2f6; padding:9px 10px; text-align:right; }}
    td:first-child, th:first-child {{ text-align:left; }}
    tfoot td {{ font-weight:700; background:#f8fafc; }}
    img {{ width:100%; display:block; border-radius:6px; }}
    .chart-card {{ padding:16px; }}
    .chart-label, .card-title {{ font-weight:700; margin-bottom:12px; }}
    .pos {{ color:var(--green); font-weight:700; }}
    .neg {{ color:var(--red); font-weight:700; }}
    .warn {{ color:var(--amber); font-weight:700; }}
    .dot, .swatch {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
    .swatch {{ width:14px; height:14px; border-radius:3px; margin-top:4px; flex:0 0 auto; }}
    .commodity {{ padding:18px; margin-bottom:18px; }}
    .commodity-title {{ display:flex; gap:10px; align-items:flex-start; margin-bottom:16px; }}
    .commodity-title p {{ margin:3px 0 0; color:var(--muted); font-size:12px; }}
    footer {{ border-top:1px solid var(--border); background:white; color:var(--muted); padding:18px 0; font-size:12px; }}
    @media (max-width: 960px) {{ .head, .stats {{ display:block; text-align:left; }} .stats {{ margin-top:18px; }} .kpis, .grid, .grid.three {{ grid-template-columns:1fr; }} .container {{ padding:0 16px; }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <div class="container head">
      <div>
        <h1>BMR / MMR</h1>
        <div class="sub">Scrapyard Portfolio Risk Report</div>
      </div>
      <div class="stats">
        <div><div class="label">Portfolio MTM</div><div class="value">{usd(total_mtm)}</div></div>
        <div><div class="label">Unrealised P&amp;L</div><div class="value {cls(total_pnl)}">{usd(total_pnl, signed=True)}</div></div>
        <div><div class="label">Report Date</div><div class="value" style="font-size:15px;color:#cbd5e1;">{report_date_long}</div></div>
      </div>
    </div>
  </header>
  <nav><div class="container">
    <a href="#overview">Portfolio</a><a href="#var">VaR</a><a href="#mc">Monte Carlo</a><a href="#stress">Stress</a><a href="#prices">Prices</a><a href="#commodities">Commodities</a><a href="#method">Methodology</a>
  </div></nav>
  <div class="notice"><div class="container">Generated {report_date_iso} from refreshed Yahoo Finance market data and current GreenSpark inventory. Charts were regenerated by <code>python run_risk_model.py --refresh</code>.</div></div>
  <main><div class="container">
    <div class="kpis">
      <div class="card kpi"><div class="label">Portfolio MTM Value</div><div class="value">{usd(total_mtm)}</div><div class="delta">{len(summary)} commodities · {total_tonnes:,.1f} tonnes held</div></div>
      <div class="card kpi"><div class="label">Unrealised Gain</div><div class="value {cls(total_pnl)}">{usd(total_pnl, signed=True)}</div><div class="delta">{pnl_pct:+.1f}% vs book cost of {usd(total_book)}</div></div>
      <div class="card kpi"><div class="label">95% VaR - 30 Day</div><div class="value warn">{usd(var95['portfolio_var'])}</div><div class="delta">{var95['portfolio_var'] / total_mtm:.1%} of portfolio · Parametric method</div></div>
      <div class="card kpi"><div class="label">P(Loss) - 30 Day MC</div><div class="value warn">{mc['metrics']['prob_loss']:.1%}</div><div class="delta">{config.MONTE_CARLO_SIMULATIONS:,} simulations · zero-drift GBM</div></div>
    </div>

    <section id="overview">
      <div class="section-head"><div><h2>Portfolio Overview</h2><div class="desc">Mark-to-market inventory valued at current spot prices and current grade-level sale prices where available.</div></div><span class="badge">Updated</span></div>
      <div class="grid">
        <div class="card span2">
          <div class="card-title">Inventory - Mark to Market</div>
          {table(inv_rows, "<tr><th>Commodity</th><th>Qty (t)</th><th>Avg Cost/t</th><th>Spot/t</th><th>Break-even/t</th><th>Basis</th><th>Book Value</th><th>MTM Value</th><th>Unr. P&L</th><th>$/t Sens.</th><th>Avg Days</th></tr>", inv_foot)}
        </div>
        {chart("output/charts/commodity_breakdown.png", "MTM Value by Commodity")}
        {chart("output/charts/inventory_aging.png", "Inventory Aging")}
      </div>
    </section>

    <section id="var">
      <div class="section-head"><div><h2>Value at Risk</h2><div class="desc">30 trading-day horizon across parametric, EWMA, and historical methods.</div></div></div>
      <div class="grid">
        {chart("output/charts/var_summary.png", "VaR Summary")}
        <div class="card"><div class="card-title">VaR Table</div>{table(var_rows, "<tr><th>Confidence</th><th>Method</th><th>VaR</th><th>CVaR</th><th>Diversif. Benefit</th></tr>")}</div>
      </div>
    </section>

    <section id="mc">
      <div class="section-head"><div><h2>Monte Carlo Simulation</h2><div class="desc">Portfolio paths over 30, 60, 90, and 180 trading-day horizons.</div></div></div>
      {chart("output/charts/monte_carlo.png", "Portfolio Value Distribution")}
      <div class="grid" style="margin-top:18px;">
        <div class="card"><div class="card-title">Percentile Ladder - 30 Day</div>{table(mc_ladder, "<tr><th>Percentile</th><th>Value</th><th>vs Today</th></tr>")}</div>
        <div class="card"><div class="card-title">Horizon Comparison</div>{table(horizon_rows, "<tr><th>Horizon</th><th>Median</th><th>5th</th><th>95th</th><th>P(Loss)</th><th>MC VaR</th><th>MC CVaR</th></tr>")}</div>
      </div>
    </section>

    <section id="stress">
      <div class="section-head"><div><h2>Stress Scenarios</h2><div class="desc">Instantaneous price shocks applied to current MTM exposures. Worst case: {esc(ws['worst_scenario'])} ({ws['worst_pnl_pct']:.1f}%). Best case: {esc(ws['best_scenario'])} ({ws['best_pnl_pct']:+.1f}%).</div></div></div>
      <div class="grid">
        {chart("output/charts/scenarios.png", "Stress Scenario P&L")}
        <div class="card"><div class="card-title">Scenario Breakdown</div>{table(scen_rows, scen_head)}</div>
      </div>
    </section>

    <section id="prices">
      <div class="section-head"><div><h2>Price History &amp; Returns</h2><div class="desc">Five-year lookback, USD per metric tonne.</div></div></div>
      <div class="grid">
        {chart("output/charts/price_history.png", "Spot Price History")}
        {chart("output/charts/return_distributions.png", "Daily Return Distributions")}
      </div>
    </section>

    <section id="commodities">
      <div class="section-head"><div><h2>Commodity Deep Dives</h2><div class="desc">Per-commodity Monte Carlo, VaR, and stress charts.</div></div></div>
      {commodity_sections}
    </section>

    <section id="method">
      <div class="section-head"><div><h2>Methodology</h2><div class="desc">Data sources and modelling conventions used in this report.</div></div></div>
      <div class="grid">
        <div class="card"><div class="card-title">Price Data Sources</div>{table(sources, "<tr><th>Commodity</th><th>Ticker / Proxy</th><th>Source</th></tr>")}</div>
        <div class="card"><div class="card-title">Risk Calculations</div>
          <table><tbody>
            <tr><td>Parametric VaR</td><td>Normal distribution, 5-year daily volatility, square-root horizon scaling.</td></tr>
            <tr><td>EWMA VaR</td><td>RiskMetrics lambda={config.EWMA_LAMBDA}, more sensitive to recent moves.</td></tr>
            <tr><td>Historical VaR</td><td>Overlapping 30-day windows from empirical returns.</td></tr>
            <tr><td>Monte Carlo</td><td>{config.MONTE_CARLO_SIMULATIONS:,} correlated GBM paths, zero-drift VaR convention.</td></tr>
            <tr><td>Mark-to-Market</td><td>Real matched market price per grade where available; otherwise futures price times basis factor.</td></tr>
          </tbody></table>
        </div>
      </div>
    </section>
  </div></main>
  <footer><div class="container">BMR / MMR Scrapyard Risk Model · Generated {report_date_long} · Internal use only.</div></footer>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(build_html(refresh=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
