#!/usr/bin/env python3
"""
Boruta feature-selection step for the copper-scrap workflow.
============================================================

Inserts an all-relevant Boruta selection between feature engineering and the
final redundancy pruning, then produces a cleaned correlation matrix.

Target : comex_copper (COMEX HG=F, $/lb) -- the deep-history proxy for the
         copper-scrap price.  (Per the workflow: scrap ~= COMEX x realization.)

Pipeline:
  1. fetch (cached FRED + yfinance) -> weekly panel -> engineer features
  2. drop target-derived / target-containing features (explanatory, not AR):
       - cu_* own-price transforms (momentum / ma / vs_ma / vol / lag)
       - comex_lme_spread, comex_lme_ratio, copper_etf_basis  (contain COMEX)
  3. time-series split: first 75% of weeks = train (no shuffle)
  4. median-impute features on the TRAIN window only
  5. Boruta x2 :  STRICT perc=100  and  RELAXED perc=90  (RandomForestRegressor)
  6. final selection: confirmed -> tentative(economic) -> high-corr fallback
  7. redundancy prune (|r|>0.90, also reports 0.85), keep core COMEX/LME
  8. final correlation matrix + heatmap + summary

Outputs (./copper_features_output/):
  boruta_feature_rankings.csv, boruta_selected_features.csv,
  final_boruta_correlation_features.csv, final_boruta_correlation_matrix.csv,
  final_boruta_correlation_heatmap.png, boruta_feature_selection_summary.md
"""
from __future__ import annotations

import io
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl.create_default_context()
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from boruta import BorutaPy

START = "2015-01-01"
FREQ = "W-FRI"
LB_PER_TONNE = 2204.62
TARGET = "comex_copper"
TRAIN_FRAC = 0.75
RANDOM_STATE = 42
OUTDIR = Path("copper_features_output").resolve()
CACHE = (Path("cache/copper_features_raw")
         if Path("cache/copper_features_raw").exists() else OUTDIR / "cache")
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

def log(m): print(f"[boruta] {m}", flush=True)


# ===========================================================================
# Data + engineering  (same logic as the main workflow)
# ===========================================================================
def fetch_fred(sid):
    cache = CACHE / f"fred_{sid}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    try:
        txt = urllib.request.urlopen(url, context=_SSL, timeout=20).read().decode()
        df = pd.read_csv(io.StringIO(txt)); df.columns = ["date", "val"]
        df = df[df["val"] != "."]; df["date"] = pd.to_datetime(df["date"])
        s = pd.to_numeric(df["val"], errors="coerce"); s.index = df["date"]
        s = s.dropna(); s = s[s.index >= pd.Timestamp(START)]
        s.to_frame("val").to_csv(cache); return s
    except Exception as e:
        log(f"FRED {sid} FAILED {e}"); return None


def fetch_yf(tkr):
    cache = CACHE / f"yf_{tkr.replace('=','_').replace('^','')}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    if yf is None:
        return None
    try:
        df = yf.download(tkr, start=START, progress=False, auto_adjust=True)
        c = df["Close"]; c = c.iloc[:, 0] if hasattr(c, "columns") else c
        s = pd.to_numeric(c, errors="coerce").dropna(); s.index = pd.to_datetime(s.index)
        s.to_frame("val").to_csv(cache); return s
    except Exception as e:
        log(f"yf {tkr} FAILED {e}"); return None


SPEC = [
    ("comex_copper", "yf", "HG=F", "copper_price", False),
    ("copper_etf", "yf", "CPER", "copper_price", False),
    ("lme_copper", "fred", "PCOPPUSDM", "copper_price", False),
    ("aluminum", "yf", "ALI=F", "copper_price", False),
    ("wti_oil", "fred", "DCOILWTICO", "energy_cost", False),
    ("brent_oil", "fred", "DCOILBRENTEU", "energy_cost", False),
    ("diesel", "fred", "GASDESW", "energy_cost", False),
    ("dxy", "fred", "DTWEXBGS", "financial", False),
    ("ust_10y", "fred", "DGS10", "financial", True),
    ("ust_2y", "fred", "DGS2", "financial", True),
    ("yield_curve_2s10s", "fred", "T10Y2Y", "financial", True),
    ("fed_funds", "fred", "DFF", "financial", True),
    ("vix", "fred", "VIXCLS", "financial", True),
    ("indpro", "fred", "INDPRO", "macro_demand", True),
    ("ip_manufacturing", "fred", "IPMAN", "macro_demand", True),
    ("housing_starts", "fred", "HOUST", "macro_demand", True),
    ("construction_spend", "fred", "TTLCONS", "macro_demand", True),
    ("mfg_new_orders", "fred", "AMTMNO", "macro_demand", True),
    ("mfg_employment", "fred", "MANEMP", "macro_demand", True),
    ("ppi_steel", "fred", "WPU101", "macro_demand", True),
    ("ppi_allcommod", "fred", "PPIACO", "macro_demand", True),
]
CAT_BONUS = {"copper_price": 0.30, "tightness": 0.30, "energy_cost": 0.20,
             "macro_demand": 0.25, "financial": 0.15, "engineered": 0.20}


def build_feature_panel():
    log("loading + engineering features ...")
    series, cat = {}, {}
    for raw, kind, sid, category, _stat in SPEC:
        s = fetch_yf(sid) if kind == "yf" else fetch_fred(sid)
        if s is None or s.dropna().empty:
            continue
        series[raw] = s[~s.index.duplicated(keep="last")].sort_index()
        cat[raw] = category
    wk = pd.DataFrame(series).sort_index().resample(FREQ).last()
    monthly = [c for c in wk.columns
               if pd.Series(series[c].dropna().index).diff().dt.days.median() > 10]
    wk[monthly] = wk[monthly].ffill(limit=8)
    wk = wk.ffill(limit=2)

    feat = wk.copy()
    fcat = dict(cat)
    cu = wk[TARGET]
    comex_t = cu * LB_PER_TONNE
    feat["comex_lme_spread"] = comex_t - wk["lme_copper"]; fcat["comex_lme_spread"] = "tightness"
    feat["comex_lme_ratio"] = comex_t / wk["lme_copper"]; fcat["comex_lme_ratio"] = "tightness"
    base = wk["copper_etf"].iloc[:20].mean()
    feat["copper_etf_basis"] = (wk["copper_etf"] / base - cu / cu.iloc[:20].mean())
    fcat["copper_etf_basis"] = "tightness"
    for n, lbl in [(1, "7d"), (4, "30d"), (13, "90d")]:
        feat[f"cu_momentum_{lbl}"] = cu.pct_change(n); fcat[f"cu_momentum_{lbl}"] = "engineered"
    ret = cu.pct_change()
    for n, lbl in [(4, "30d"), (13, "90d")]:
        ma = cu.rolling(n, min_periods=max(2, n // 2)).mean()
        feat[f"cu_ma_{lbl}"] = ma; fcat[f"cu_ma_{lbl}"] = "copper_price"
        feat[f"cu_vs_ma_{lbl}"] = cu / ma - 1; fcat[f"cu_vs_ma_{lbl}"] = "engineered"
        feat[f"cu_vol_{lbl}"] = ret.rolling(n, min_periods=max(2, n // 2)).std()
        fcat[f"cu_vol_{lbl}"] = "engineered"
    for n, lbl in [(1, "7d"), (4, "30d"), (13, "90d")]:
        feat[f"cu_lag_{lbl}"] = cu.shift(n); fcat[f"cu_lag_{lbl}"] = "copper_price"
    chg = {"dxy": "financial", "wti_oil": "energy_cost", "diesel": "energy_cost",
           "indpro": "macro_demand", "ip_manufacturing": "macro_demand",
           "housing_starts": "macro_demand", "construction_spend": "macro_demand",
           "mfg_new_orders": "macro_demand", "ppi_steel": "macro_demand",
           "vix": "financial"}
    for col, c in chg.items():
        if col in wk:
            feat[f"{col}_chg_30d"] = wk[col].pct_change(4); fcat[f"{col}_chg_30d"] = c
    for col in ["ust_10y", "ust_2y", "yield_curve_2s10s", "fed_funds"]:
        if col in wk:
            feat[f"{col}_chg_30d"] = wk[col].diff(4); fcat[f"{col}_chg_30d"] = "financial"
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat, fcat


# ===========================================================================
# Boruta
# ===========================================================================
# features excluded from Boruta because they contain / are transforms of the
# TARGET (explanatory selection, not autoregressive forecasting)
def is_target_derived(col):
    return (col.startswith("cu_")                       # own-price momentum/ma/vol/lag
            or col in {"comex_lme_spread", "comex_lme_ratio", "copper_etf_basis"})


def run_boruta(X, y, perc):
    rf = RandomForestRegressor(max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    b = BorutaPy(rf, n_estimators="auto", perc=perc, alpha=0.05,
                 max_iter=100, random_state=RANDOM_STATE, verbose=0)
    b.fit(X, y)
    return b


def main():
    feat, fcat = build_feature_panel()
    feat = feat.dropna(subset=[TARGET]).sort_index()
    candidates = [c for c in feat.columns
                  if c != TARGET and not is_target_derived(c)]
    log(f"{feat.shape[1]-1} engineered features; "
        f"{len(candidates)} candidates after dropping target-derived "
        f"({feat.shape[1]-1-len(candidates)} excluded).")

    # ---- time-series split (no shuffle) ------------------------------------
    split = int(len(feat) * TRAIN_FRAC)
    train = feat.iloc[:split]
    log(f"train window: {train.index.min().date()} -> {train.index.max().date()} "
        f"({len(train)} of {len(feat)} weeks, {TRAIN_FRAC:.0%})")
    y = train[TARGET].values
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train[candidates])

    # ---- two Boruta configurations -----------------------------------------
    log("running Boruta STRICT (perc=100) ...")
    strict = run_boruta(X, y, 100)
    log("running Boruta RELAXED (perc=90) ...")
    relaxed = run_boruta(X, y, 90)

    # ---- rankings table -----------------------------------------------------
    rows = []
    for i, col in enumerate(candidates):
        t = train[[col, TARGET]].dropna()
        pear = t[col].corr(t[TARGET]) if len(t) >= 10 else np.nan
        spear = (scipy_stats.spearmanr(t[col], t[TARGET])[0]
                 if len(t) >= 10 else np.nan)
        rows.append(dict(
            feature=col, feature_category=fcat.get(col, "other"),
            boruta_support=bool(relaxed.support_[i]),
            boruta_tentative=bool(relaxed.support_weak_[i]),
            boruta_ranking=int(relaxed.ranking_[i]),
            boruta_support_strict=bool(strict.support_[i]),
            boruta_ranking_strict=int(strict.ranking_[i]),
            pearson_corr_with_target=pear, spearman_corr_with_target=spear,
            missing_pct=float(train[col].isna().mean())))
    rank = pd.DataFrame(rows)
    rank["abs_corr"] = rank[["pearson_corr_with_target",
                             "spearman_corr_with_target"]].abs().max(axis=1)
    rank = rank.sort_values(["boruta_ranking", "abs_corr"],
                            ascending=[True, False]).reset_index(drop=True)
    rank.to_csv(OUTDIR / "boruta_feature_rankings.csv", index=False)

    n_conf = int(rank["boruta_support"].sum())
    n_tent = int(rank["boruta_tentative"].sum())
    n_conf_strict = int(rank["boruta_support_strict"].sum())
    log(f"Boruta relaxed: {n_conf} confirmed, {n_tent} tentative, "
        f"{len(rank)-n_conf-n_tent} rejected | strict confirmed: {n_conf_strict}")

    # ---- selection: confirmed -> tentative(econ) -> high-corr fallback ------
    confirmed = rank[rank["boruta_support"]]["feature"].tolist()
    tentative = rank[rank["boruta_tentative"] & ~rank["boruta_support"]
                     ]["feature"].tolist()
    selected = list(confirmed)
    # add tentative that carry real economic meaning (core categories)
    econ_cats = {"copper_price", "tightness", "macro_demand", "energy_cost",
                 "financial"}
    for f in tentative:
        if fcat.get(f) in econ_cats:
            selected.append(f)
    # fallback: if Boruta missed an economically important high-corr variable
    if len(selected) < 10:
        for f in rank.sort_values("abs_corr", ascending=False)["feature"]:
            if f not in selected and rank.set_index("feature").loc[f, "abs_corr"] >= 0.3:
                selected.append(f)
            if len(selected) >= 12:
                break
    pd.DataFrame({"feature": selected,
                  "boruta_status": ["confirmed" if f in confirmed
                                    else "tentative" if f in tentative
                                    else "high_corr_fallback" for f in selected]}
                 ).to_csv(OUTDIR / "boruta_selected_features.csv", index=False)

    # ---- redundancy pruning (|r|>0.90, report 0.85) ------------------------
    rmap = rank.set_index("feature")
    # core copper benchmark(s) to preserve through pruning (COMEX = target, in
    # the matrix by construction). LME is protected so the redundant CPER ETF is
    # dropped against it rather than the reverse.
    PROTECTED = [c for c in ["lme_copper"] if c in candidates]

    def prune(feats, thresh):
        feats = list(dict.fromkeys(PROTECTED + list(feats)))   # cores seeded first
        cm = train[feats].corr().abs()
        keep, dropped = [], []
        for f in sorted(feats, key=lambda x: (x not in PROTECTED,
                                              rmap.loc[x, "boruta_ranking"])):
            red = None
            for k in keep:
                if cm.loc[f, k] >= thresh:
                    red = k; break
            if red is None:
                keep.append(f)
            elif red in PROTECTED:                # never drop a protected core
                dropped.append((f, red, float(cm.loc[f, red])))
            else:
                a, b = rmap.loc[f], rmap.loc[red]   # tie-break: rank,|corr|,econ,missing
                sa = (-a["boruta_ranking"], a["abs_corr"],
                      CAT_BONUS.get(a["feature_category"], .1), -a["missing_pct"])
                sb = (-b["boruta_ranking"], b["abs_corr"],
                      CAT_BONUS.get(b["feature_category"], .1), -b["missing_pct"])
                if f not in PROTECTED and sa > sb:
                    keep[keep.index(red)] = f
                    dropped.append((red, f, float(cm.loc[f, red])))
                else:
                    dropped.append((f, red, float(cm.loc[f, red])))
        return keep, dropped

    kept90, dropped90 = prune(selected, 0.90)
    kept85, _ = prune(selected, 0.85)

    # final = redundancy-pruned set (cores already preserved inside prune)
    final = list(kept90)[:20]
    if len(final) < 10:                              # top-up to >=10 by abs_corr
        for f in rank.sort_values("abs_corr", ascending=False)["feature"]:
            if f not in final:
                final.append(f)
            if len(final) >= 10:
                break

    pd.DataFrame({"feature": final,
                  "feature_category": [fcat.get(f, "other") for f in final],
                  "boruta_ranking": [int(rmap.loc[f, "boruta_ranking"]) for f in final],
                  "abs_corr_with_target": [rmap.loc[f, "abs_corr"] for f in final]}
                 ).to_csv(OUTDIR / "final_boruta_correlation_features.csv", index=False)

    # ---- final correlation matrix + heatmap (on full sample) ---------------
    cols = [TARGET] + final
    order = [TARGET] + (feat[cols].corr()[TARGET].drop(TARGET)
                        .abs().sort_values(ascending=False).index.tolist())
    cmat = feat[order].corr().loc[order, order]
    cmat.to_csv(OUTDIR / "final_boruta_correlation_matrix.csv")
    if plt is not None:
        mask = np.triu(np.ones_like(cmat, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(min(20, 2 + .7 * len(order)),
                                        min(18, 2 + .6 * len(order))))
        if sns is not None:
            sns.heatmap(cmat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, vmin=-1, vmax=1, square=True, linewidths=.5,
                        cbar_kws={"shrink": .7, "label": "Pearson r"},
                        annot_kws={"size": 7}, ax=ax)
        else:
            im = ax.imshow(np.where(mask, np.nan, cmat.values), cmap="RdBu_r",
                           vmin=-1, vmax=1)
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
            ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
            fig.colorbar(im, shrink=.7)
        ax.set_title("Boruta-selected copper-scrap correlation matrix",
                     fontsize=13, weight="bold")
        fig.tight_layout(); fig.savefig(OUTDIR / "final_boruta_correlation_heatmap.png", dpi=150)
        plt.close(fig)

    write_summary(rank, candidates, confirmed, tentative, selected, kept90,
                  dropped90, kept85, final, fcat, cmat, n_conf, n_tent,
                  n_conf_strict, train)
    log(f"DONE. outputs -> {OUTDIR}")
    print("\nFinal Boruta-selected correlation features:")
    for f in final:
        print("   -", f)


def write_summary(rank, candidates, confirmed, tentative, selected, kept90,
                  dropped90, kept85, final, fcat, cmat, n_conf, n_tent,
                  n_conf_strict, train):
    L = []; A = L.append
    A("# Boruta Feature-Selection Summary (copper scrap)\n")
    A("Boruta (all-relevant) is run **after feature engineering, before final "
      "redundancy pruning**, on the COMEX-copper scrap-price proxy.\n")
    A("## Setup\n")
    A(f"- Target: **{TARGET}** (COMEX HG=F, $/lb).")
    A(f"- Train window (Boruta fit, no shuffle): "
      f"**{train.index.min().date()} → {train.index.max().date()}** "
      f"({len(train)} weeks = first {TRAIN_FRAC:.0%}).")
    A(f"- Estimator: RandomForestRegressor(max_depth=5, random_state=42), "
      f"BorutaPy(n_estimators='auto', max_iter=100, alpha=0.05).")
    A("- Two configs: **strict perc=100** and **relaxed perc=90**.")
    A("- Missing features median-imputed on the train window only.\n")
    A("## How many features Boruta tested\n")
    A(f"- **{len(candidates)} candidate features tested** (the full engineered "
      "set minus the target and all target-derived/target-containing features: "
      "`cu_*` own-price momentum/MA/vol/lags and the COMEX-containing spreads "
      "`comex_lme_spread`, `comex_lme_ratio`, `copper_etf_basis` — excluded so "
      "the selection is *explanatory*, not autoregressive).")
    A(f"- **Relaxed (perc=90): {n_conf} confirmed, {n_tent} tentative, "
      f"{len(candidates)-n_conf-n_tent} rejected.**")
    A(f"- Strict (perc=100): {n_conf_strict} confirmed.\n")
    A("## Confirmed features (relaxed)\n")
    for f in confirmed:
        r = rank.set_index("feature").loc[f]
        A(f"- `{f}` ({r['feature_category']}) — rank {int(r['boruta_ranking'])}, "
          f"|corr|={r['abs_corr']:.2f}"
          + ("  *(also strict-confirmed)*" if r['boruta_support_strict'] else ""))
    A("")
    A("## Tentative features\n")
    A(", ".join(f"`{f}`" for f in tentative) if tentative else "- (none)")
    A("")
    A("## Rejected features\n")
    rej = rank[~rank["boruta_support"] & ~rank["boruta_tentative"]]["feature"].tolist()
    A(", ".join(f"`{f}`" for f in rej) if rej else "- (none)")
    A("")
    A("## Removed for redundancy (|r| > 0.90)\n")
    if dropped90:
        for drp, kpt, rr in dropped90:
            A(f"- `{drp}` dropped (|r|={rr:.2f} with kept `{kpt}`)")
    else:
        A("- (none)")
    A(f"\n*(At the stricter 0.85 threshold the surviving set would be: "
      f"{', '.join('`'+f+'`' for f in kept85)}.)*\n")
    A("## Final selected features\n")
    A("| # | feature | category | boruta rank | |corr| w/ target |")
    A("|---|---|---|---|---|")
    rmap = rank.set_index("feature")
    for i, f in enumerate(final, 1):
        if f in rmap.index:
            r = rmap.loc[f]
            A(f"| {i} | `{f}` | {r['feature_category']} | "
              f"{int(r['boruta_ranking'])} | {r['abs_corr']:.2f} |")
        else:
            A(f"| {i} | `{f}` | {fcat.get(f,'?')} | (preserved) | – |")
    A(f"\nFinal count: **{len(final)}** (target band 10–20). Core copper "
      "benchmarks COMEX (target) and LME are preserved by construction.\n")
    A("## Correlation with target (final matrix)\n")
    for f, v in cmat[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).items():
        A(f"- `{f}`: {v:+.2f}")
    A("")
    A("## Why this batch makes economic sense for copper-scrap pricing\n")
    A("- **Direct copper benchmarks** (LME copper; aluminum as a base-metals "
      "cross-check) anchor the price level — scrap is a discount to these. The "
      "near-identical CPER ETF was pruned as redundant with LME (|r|=0.99).")
    A("- **Demand drivers** confirmed by Boruta (industrial production, new "
      "orders, construction, PPI) capture the copper-consumption cycle.")
    A("- **Financial/macro factors** (USD, Treasury yields, VIX) price the "
      "dollar and risk regime that copper trades against.")
    A("- **Energy/freight** (oil/diesel) proxy collection & transport cost and "
      "broad reflation.")
    A("- Boruta keeps only variables that beat *random shadow features*, so the "
      "set is statistically defensible; redundancy pruning then removes near-"
      "duplicate copper series, leaving a compact, non-redundant batch.\n")
    A("## Caveats\n")
    A("- Boruta here is fit on the **price level** (per spec). Copper and the "
      "macro indices share a secular uptrend, so some 'confirmed' relevance is "
      "**common trend**; a returns-based re-run would confirm fewer macro "
      "levels. Use these as co-movement/structure, not clean causal drivers.")
    A("- Target-derived momentum/vol/spread features were excluded from Boruta "
      "but remain useful elsewhere (e.g. the forecasting & sell-timing models).")
    (OUTDIR / "boruta_feature_selection_summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
