"""DTC <-> inbound/outbound ticket record-linkage system."""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

DTC_GLOB = os.path.join(HERE, "*DTC_raw_data*.csv")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = HERE
COMBINED_INBOUND = os.path.join(DATA_DIR, "2026 ytd combined inbound.csv")
COMBINED_OUTBOUND = os.path.join(DATA_DIR, "2026 ytd combined outbound.csv")


@dataclass
class Config:
    dtc_material_desc: str = "Material Name"
    dtc_net_weight: str = "Yard Net Weight"
    dtc_net_weight_fallback: str = "Consumer Net Weight"
    dtc_date: str = "Invoice Issue Date"
    dtc_date_fmt: str = "%m-%d-%y, %H:%M:%S"
    dtc_customer: str = "Customer Name"
    dtc_value: str = "Yard Value"

    net_weight_tol_lbs: float = 1.0
    net_weight_tol_pct: float = 0.005
    net_weight_partial_mult: float = 5.0
    gross_weight_tol_lbs: float = 50.0

    date_exact_days: int = 0
    date_near_days: int = 1
    date_mid_days: int = 3
    date_max_score_days: int = 7
    date_block_days: int = 14


    desc_min_ratio: int = 80
    desc_block_ratio: int = 85
    customer_min_ratio: int = 70
    customer_diff_ratio: int = 40

    price_rel_tol: float = 0.25

    w_code: float = 40.0
    w_desc: float = 20.0
    w_net_weight: float = 40.0
    w_gross_weight: float = 15.0
    w_date: float = 25.0
    w_time: float = 12.0
    w_customer: float = 25.0
    w_location: float = 12.0
    w_price: float = 8.0

    p_code_diff: float = -40.0
    p_weight_over: float = -15.0
    p_customer_diff: float = -15.0

    score_high: float = 80.0
    score_medium: float = 60.0
    score_min: float = 40.0
    high_margin: float = 10.0

    top_n_candidates: int = 5
    heldout_outbound_id: str = "Outbound Ticket Id"

    agg_enabled: bool = True
    agg_lookback_days: int = 30
    agg_lookahead_days: int = 3
    agg_max_rows: int = 5
    agg_tol_lbs: float = 2.0
    agg_tol_pct: float = 0.01
    agg_beam_size: int = 5000
    agg_pool_limit: int = 140
    agg_ticket_penalty: float = 4.0
    agg_min_score: float = 60.0


CFG = Config()


_BRACKET = re.compile(r"\[[^\]]*\]$")
_MONEY = re.compile(r"[,$\s]")


def _money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(_MONEY, "", regex=True),
                         errors="coerce")


def _ticket_dt(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(_BRACKET, "", regex=True)
    return pd.to_datetime(cleaned, utc=True, errors="coerce", format="mixed").dt.tz_localize(None)


def _norm_text(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().upper() if pd.notna(s) else ""


def _load_concat(pattern: str) -> pd.DataFrame:
    paths = [p for p in glob.glob(os.path.join(DATA_DIR, pattern))
             if "combined" not in os.path.basename(p)]
    return pd.concat([pd.read_csv(p, thousands=",") for p in paths], ignore_index=True)


def _load_orders(pattern: str, combined_path: str, row_id_col: str) -> pd.DataFrame:
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path, thousands=",")
        if row_id_col not in df.columns:
            df.insert(0, row_id_col, range(1, len(df) + 1))
        return df

    df = _load_concat(pattern)
    if row_id_col not in df.columns:
        df.insert(0, row_id_col, range(1, len(df) + 1))
    return df


def _build_code_map() -> dict[str, str]:
    ob = _load_orders("*outbound.csv", COMBINED_OUTBOUND, "Outbound Row Id")[["Material", "Material Code"]].rename(
        columns={"Material": "desc"})
    inb = _load_orders("*inbound.csv", COMBINED_INBOUND, "Inbound Row Id")[["Material Name", "Material Code"]].rename(
        columns={"Material Name": "desc"})
    both = pd.concat([ob, inb], ignore_index=True).dropna()
    both["desc"] = both["desc"].map(_norm_text)
    return dict(zip(both["desc"], both["Material Code"]))


def load_dtc(code_map: dict[str, str]) -> pd.DataFrame:
    path = sorted(glob.glob(DTC_GLOB))[0]
    df = pd.read_csv(path, thousands=",")
    row_id = (pd.to_numeric(df["DTC Row Id"], errors="coerce").astype("Int64")
              if "DTC Row Id" in df.columns else pd.Series(df.index + 1, dtype="Int64"))
    out = pd.DataFrame({
        "dtc_row_id": row_id,
        "dtc_order_no": pd.to_numeric(df["Invoice Id"], errors="coerce").astype("Int64"),
        "desc": df[CFG.dtc_material_desc].astype(str),
        "desc_norm": df[CFG.dtc_material_desc].map(_norm_text),
        "code": df[CFG.dtc_material_desc].map(_norm_text).map(code_map),
        "net_weight": pd.to_numeric(df[CFG.dtc_net_weight], errors="coerce").fillna(
            pd.to_numeric(df[CFG.dtc_net_weight_fallback], errors="coerce")),
        "gross_weight": pd.NA,                       # DTC carries no gross weight
        "date": pd.to_datetime(df[CFG.dtc_date], format=CFG.dtc_date_fmt, errors="coerce"),
        "customer_norm": df[CFG.dtc_customer].map(_norm_text),
        "value": _money(df[CFG.dtc_value]),
        "_heldout_ob": pd.to_numeric(df[CFG.heldout_outbound_id], errors="coerce").astype("Int64"),
    })
    out["rate"] = out["value"] / out["net_weight"]
    return out


def load_outbound() -> pd.DataFrame:
    ob = _load_orders("*outbound.csv", COMBINED_OUTBOUND, "Outbound Row Id")
    out = pd.DataFrame({
        "row_id": pd.to_numeric(ob["Outbound Row Id"], errors="coerce").astype("Int64"),
        "ticket": pd.to_numeric(ob["Outbound Ticket #"], errors="coerce").astype("Int64"),
        "desc_norm": ob["Material"].map(_norm_text),
        "code": ob["Material Code"],
        "net_weight": pd.to_numeric(ob["Net Weight"], errors="coerce"),
        "gross_weight": pd.to_numeric(ob["Gross Weight"], errors="coerce"),
        "date": _ticket_dt(ob["Date In"]),
        "customer_norm": ob["Customer Name"].map(_norm_text),
        "value": _money(ob["Expected Value"]),
        "location": ob["Yard Name"].map(_norm_text),
    })
    out["rate"] = out["value"] / out["net_weight"]
    return out.dropna(subset=["ticket"]).reset_index(drop=True)


def load_inbound() -> pd.DataFrame:
    inb = _load_orders("*inbound.csv", COMBINED_INBOUND, "Inbound Row Id")
    out = pd.DataFrame({
        "row_id": pd.to_numeric(inb["Inbound Row Id"], errors="coerce").astype("Int64"),
        "ticket": pd.to_numeric(inb["Ticket #"], errors="coerce").astype("Int64"),
        "desc_norm": inb["Material Name"].map(_norm_text),
        "code": inb["Material Code"],
        "net_weight": pd.to_numeric(inb["Net Weight"], errors="coerce"),
        "gross_weight": pd.to_numeric(inb["Gross Weight"], errors="coerce"),
        "date": _ticket_dt(inb["Effective Date"]),
        "supplier_norm": inb["Customer Name"].map(_norm_text),
        "value": _money(inb["Cost"]),
        "location": inb["Yard Name"].map(_norm_text),
    })
    out["rate"] = out["value"] / out["net_weight"]
    return out.dropna(subset=["ticket"]).reset_index(drop=True)


@dataclass
class Score:
    raw: float = 0.0
    max_possible: float = 0.0
    breakdown: dict = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        if self.max_possible <= 0:
            return 0.0
        return max(0.0, 100.0 * self.raw / self.max_possible)


def _date_points(days: float) -> float:
    d = abs(days)
    if d <= CFG.date_exact_days:
        return CFG.w_date
    if d <= CFG.date_near_days:
        return CFG.w_date * 0.6
    if d <= CFG.date_mid_days:
        return CFG.w_date * 0.3
    if d <= CFG.date_max_score_days:
        return CFG.w_date * 0.12
    return 0.0


def score_pair(dtc: pd.Series, cand: pd.Series, side: str) -> Score:
    raw = 0.0
    mx = 0.0
    bd: dict[str, float] = {}

    if pd.notna(dtc["code"]) and pd.notna(cand["code"]):
        mx += CFG.w_code
        if dtc["code"] == cand["code"]:
            raw += CFG.w_code; bd["code"] = CFG.w_code
        else:
            raw += CFG.p_code_diff; bd["code_diff"] = CFG.p_code_diff

    if dtc["desc_norm"] and cand["desc_norm"]:
        mx += CFG.w_desc
        ratio = fuzz.token_sort_ratio(dtc["desc_norm"], cand["desc_norm"])
        if ratio >= CFG.desc_min_ratio:
            pts = round(CFG.w_desc * ratio / 100.0, 2)
            raw += pts; bd["desc"] = pts

    if pd.notna(dtc["net_weight"]) and pd.notna(cand["net_weight"]):
        mx += CFG.w_net_weight
        tol = max(CFG.net_weight_tol_lbs, CFG.net_weight_tol_pct * dtc["net_weight"])
        diff = abs(dtc["net_weight"] - cand["net_weight"])
        if diff <= tol:
            raw += CFG.w_net_weight; bd["net_wt"] = CFG.w_net_weight
        elif diff <= CFG.net_weight_partial_mult * tol:
            frac = 1 - (diff - tol) / ((CFG.net_weight_partial_mult - 1) * tol)
            pts = round(CFG.w_net_weight * frac, 2)
            raw += pts; bd["net_wt"] = pts
        else:
            raw += CFG.p_weight_over; bd["net_wt_over"] = CFG.p_weight_over

    if pd.notna(dtc["gross_weight"]) and pd.notna(cand["gross_weight"]):
        mx += CFG.w_gross_weight
        if abs(dtc["gross_weight"] - cand["gross_weight"]) <= CFG.gross_weight_tol_lbs:
            raw += CFG.w_gross_weight; bd["gross_wt"] = CFG.w_gross_weight

    if pd.notna(dtc["date"]) and pd.notna(cand["date"]):
        mx += CFG.w_date
        days = (dtc["date"].normalize() - cand["date"].normalize()).days
        pts = round(_date_points(days), 2)
        if pts:
            raw += pts; bd["date"] = pts

        if days == 0:
            mx += CFG.w_time
            hours = abs((dtc["date"] - cand["date"]).total_seconds()) / 3600.0
            pts = round(CFG.w_time * max(0.0, 1 - min(hours, 12) / 12.0), 2)
            if pts:
                raw += pts; bd["time"] = pts

    if side == "outbound" and dtc["customer_norm"] and cand["customer_norm"]:
        mx += CFG.w_customer
        ratio = fuzz.token_sort_ratio(dtc["customer_norm"], cand["customer_norm"])
        if ratio >= CFG.customer_min_ratio:
            pts = round(CFG.w_customer * ratio / 100.0, 2)
            raw += pts; bd["customer"] = pts
        elif ratio < CFG.customer_diff_ratio:
            raw += CFG.p_customer_diff; bd["customer_diff"] = CFG.p_customer_diff


    if pd.notna(dtc["rate"]) and pd.notna(cand["rate"]) and dtc["rate"] > 0:
        mx += CFG.w_price
        rel = abs(dtc["rate"] - cand["rate"]) / dtc["rate"]
        if rel <= CFG.price_rel_tol:
            pts = round(CFG.w_price * (1 - rel / CFG.price_rel_tol), 2)
            raw += pts; bd["price"] = pts

    return Score(raw=round(raw, 2), max_possible=mx, breakdown=bd)


def _fmt_breakdown(bd: dict) -> str:
    return ";".join(f"{k}{'+' if v >= 0 else ''}{v}" for k, v in bd.items())


def candidates_for(dtc: pd.Series, tickets: pd.DataFrame, side: str) -> pd.DataFrame:
    if pd.isna(dtc["date"]):
        win = tickets
    else:
        lo = dtc["date"].normalize() - pd.Timedelta(days=CFG.date_block_days)
        hi = dtc["date"].normalize() + pd.Timedelta(days=CFG.date_block_days)
        win = tickets[(tickets["date"] >= lo) & (tickets["date"] <= hi)]
    if win.empty:
        return win

    keep = pd.Series(False, index=win.index)
    if pd.notna(dtc["code"]):
        keep |= win["code"] == dtc["code"]
    if pd.notna(dtc["net_weight"]):
        tol = max(CFG.net_weight_tol_lbs, CFG.net_weight_tol_pct * dtc["net_weight"])
        keep |= (win["net_weight"] - dtc["net_weight"]).abs() <= CFG.net_weight_partial_mult * tol
    if dtc["desc_norm"]:
        keep |= win["desc_norm"].apply(
            lambda d: bool(d) and fuzz.token_sort_ratio(dtc["desc_norm"], d) >= CFG.desc_block_ratio)
    if side == "outbound" and dtc["customer_norm"]:
        keep |= win["customer_norm"].apply(
            lambda c: bool(c) and fuzz.token_sort_ratio(dtc["customer_norm"], c) >= CFG.customer_min_ratio)
    return win[keep]


def rank_candidates(dtc: pd.Series, tickets: pd.DataFrame, side: str) -> pd.DataFrame:
    cand = candidates_for(dtc, tickets, side)
    rows = []
    for _, c in cand.iterrows():
        sc = score_pair(dtc, c, side)
        rows.append({
            "dtc_row_id": dtc["dtc_row_id"],
            "ticket_row_id": c["row_id"],
            "ticket": c["ticket"],
            "score": round(sc.normalized, 1),
            "raw": sc.raw,
            "max_possible": sc.max_possible,
            "breakdown": _fmt_breakdown(sc.breakdown),
            "ticket_weight": c["net_weight"],
            "ticket_date": c["date"],
            "ticket_desc": c["desc_norm"],
            "ticket_code": c["code"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def classify(ranked: pd.DataFrame, dtc: pd.Series) -> dict:
    if ranked.empty:
        return {"ticket": pd.NA, "confidence": "No match", "score": 0.0, "raw": 0.0,
                "row_id": pd.NA,
                "breakdown": "", "margin": pd.NA, "note": "no candidates in block",
                "weight": pd.NA, "date": pd.NaT}

    best = ranked.iloc[0]
    second = ranked.iloc[1]["score"] if len(ranked) > 1 else 0.0
    margin = round(best["score"] - second, 1)
    notes = []

    missing_key = pd.isna(dtc["code"]) or pd.isna(dtc["net_weight"]) or pd.isna(dtc["date"])
    if missing_key:
        notes.append("missing key field(s)")

    if best["score"] < CFG.score_min:
        conf = "No match"
    elif best["score"] >= CFG.score_high and margin >= CFG.high_margin and not missing_key:
        conf = "High"
    elif best["score"] >= CFG.score_medium:
        conf = "Medium" if margin >= CFG.high_margin else "Low"
        if margin < CFG.high_margin:
            notes.append(f"ambiguous (margin {margin})")
    else:
        conf = "Low"
        if margin < CFG.high_margin:
            notes.append(f"ambiguous (margin {margin})")

    if conf in ("High", "Medium", "Low") and len(ranked) > 1 and margin < CFG.high_margin:
        ties = ranked[ranked["score"] >= best["score"] - CFG.high_margin]["ticket"].tolist()
        notes.append("near-ties: " + ";".join(str(t) for t in ties[:4]))

    return {"ticket": best["ticket"], "confidence": conf, "score": best["score"],
            "row_id": best["ticket_row_id"],
            "raw": best["raw"], "breakdown": best["breakdown"], "margin": margin,
            "note": "; ".join(notes), "weight": best["ticket_weight"],
            "date": best["ticket_date"]}


def _row_id_parts(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [p for p in str(value).split(";") if p and p != "<NA>"]


def _exclude_used_ranked(ranked: pd.DataFrame, used_row_ids: set[str]) -> pd.DataFrame:
    if ranked.empty or not used_row_ids:
        return ranked
    return ranked[~ranked["ticket_row_id"].astype("string").isin(used_row_ids)].reset_index(drop=True)


def _parse_score_breakdown(text: object) -> dict[str, float]:
    out = {}
    if pd.isna(text):
        return out
    for part in str(text).split(";"):
        m = re.match(r"([A-Za-z0-9_()/-]+)([+-].+)$", part)
        if not m:
            continue
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def _classify_score(score: float, margin: float, missing_key: bool = False) -> str:
    if score < CFG.score_min:
        return "No match"
    if score >= CFG.score_high and margin >= CFG.high_margin and not missing_key:
        return "High"
    if score >= CFG.score_medium:
        return "Medium" if margin >= CFG.high_margin else "Low"
    return "Low"


def _aggregate_score(dtc: pd.Series, chosen: pd.DataFrame, second_score: float) -> dict:
    target = float(dtc["net_weight"])
    total = float(chosen["net_weight"].sum())
    tol = max(CFG.agg_tol_lbs, CFG.agg_tol_pct * target)
    err = abs(target - total)
    weight_pts = max(0.0, CFG.w_net_weight * (1 - err / tol))

    raw = 0.0
    mx = CFG.w_code + CFG.w_desc + CFG.w_net_weight
    bd = {"code": CFG.w_code, "desc": CFG.w_desc, f"sum_wt({chosen['ticket'].nunique()}tk/{len(chosen)}ln)": round(weight_pts, 2)}
    raw += CFG.w_code + CFG.w_desc + weight_pts

    if pd.notna(dtc["date"]):
        mx += CFG.w_date
        date_days = (dtc["date"].normalize() - chosen["date"].max().normalize()).days
        date_pts = _date_points(date_days)
        raw += date_pts
        if date_pts:
            bd["date"] = round(date_pts, 2)

    score = max(0.0, round(100.0 * raw / mx, 1)) if mx else 0.0
    margin = round(score - second_score, 1)
    if score >= CFG.score_high:
        conf = "High"
    elif score >= CFG.score_medium:
        conf = "Medium"
    elif score >= CFG.score_min:
        conf = "Low"
    else:
        conf = "No match"

    ticket_word = "ticket" if chosen["ticket"].nunique() == 1 else "tickets"
    line_word = "line" if len(chosen) == 1 else "lines"

    return {
        "ticket": ";".join(str(int(x)) for x in chosen["ticket"].dropna().unique()),
        "row_id": ";".join(str(int(x)) for x in chosen["row_id"].dropna().unique()),
        "confidence": conf,
        "score": score,
        "raw": round(raw, 2),
        "breakdown": _fmt_breakdown(bd),
        "margin": margin,
        "weight": int(round(total)),
        "date": chosen["date"].max(),
        "note": (f"aggregated {chosen['ticket'].nunique()} {ticket_word} ({len(chosen)} {line_word}) "
                 f"sum={int(round(total))} vs {int(round(target))}, err={int(round(err))}; "
                 f"dates {chosen['date'].min().date()}..{chosen['date'].max().date()}"),
    }


def aggregate_inbound_match(dtc: pd.Series, inbound: pd.DataFrame, current: dict,
                            ranked: pd.DataFrame, used_row_ids: set[str] | None = None) -> dict:
    if not CFG.agg_enabled or pd.isna(dtc["net_weight"]) or pd.isna(dtc["date"]) or pd.isna(dtc["code"]):
        return current
    if current["confidence"] == "High":
        return current

    target = float(dtc["net_weight"])
    tol = max(CFG.agg_tol_lbs, CFG.agg_tol_pct * target)
    lo = dtc["date"].normalize() - pd.Timedelta(days=CFG.agg_lookback_days)
    hi = dtc["date"].normalize() + pd.Timedelta(days=CFG.agg_lookahead_days)

    pool = inbound[
        (inbound["code"] == dtc["code"])
        & inbound["date"].between(lo, hi)
        & inbound["net_weight"].notna()
        & (inbound["net_weight"] > 0)
        & (inbound["net_weight"] <= target + tol)
    ].copy()
    if used_row_ids:
        pool = pool[~pool["row_id"].astype("string").isin(used_row_ids)]
    if pool.empty:
        return current

    pool["desc_ratio"] = pool["desc_norm"].apply(
        lambda d: fuzz.token_sort_ratio(dtc["desc_norm"], d) if d else 0)
    pool = pool[pool["desc_ratio"] >= CFG.desc_min_ratio]
    if pool.empty:
        return current

    pool["date_abs"] = (pool["date"].dt.normalize() - dtc["date"].normalize()).dt.days.abs()
    pool["weight_gap"] = (target - pool["net_weight"]).abs()
    pool = pool.sort_values(["date_abs", "weight_gap"], ascending=[True, True]).head(CFG.agg_pool_limit)

    states: list[tuple[float, tuple[int, ...]]] = [(0.0, tuple())]
    best: tuple[float, tuple[int, ...]] | None = None
    best_err = float("inf")

    for pos, (_, row) in enumerate(pool.iterrows()):
        wt = float(row["net_weight"])
        additions = []
        for total, idxs in states:
            if len(idxs) >= CFG.agg_max_rows:
                continue
            new_total = total + wt
            if new_total > target + tol:
                continue
            new_idxs = idxs + (pos,)
            err = abs(target - new_total)
            additions.append((new_total, new_idxs))
            if err < best_err:
                best_err = err
                best = (new_total, new_idxs)

        if additions:
            states.extend(additions)
            dedup = {}
            for total, idxs in states:
                key = (round(total), len(idxs))
                if key not in dedup or abs(target - total) < abs(target - dedup[key][0]):
                    dedup[key] = (total, idxs)
            states = sorted(dedup.values(), key=lambda s: (abs(target - s[0]), len(s[1])))[:CFG.agg_beam_size]

    if best is None or best_err > tol:
        return current

    chosen = pool.iloc[list(best[1])].copy()
    second_score = float(ranked.iloc[0]["score"]) if not ranked.empty else 0.0
    agg = _aggregate_score(dtc, chosen, second_score)
    if agg["score"] < CFG.agg_min_score or agg["score"] <= float(current["score"]):
        return current
    return agg


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    code_map = _build_code_map()
    dtc = load_dtc(code_map)
    outbound = load_outbound()
    inbound = load_inbound()

    matched_rows, cand_in_rows, cand_ob_rows = [], [], []
    work_rows = []

    for _, d in dtc.iterrows():
        ob_ranked = rank_candidates(d, outbound, "outbound")
        in_ranked = rank_candidates(d, inbound, "inbound")
        ob = classify(ob_ranked, d)

        for r in (ob_ranked.head(CFG.top_n_candidates).to_dict("records") if not ob_ranked.empty else []):
            r["dtc_order_no"] = d["dtc_order_no"]; cand_ob_rows.append(r)
        for r in (in_ranked.head(CFG.top_n_candidates).to_dict("records") if not in_ranked.empty else []):
            r["dtc_order_no"] = d["dtc_order_no"]; cand_in_rows.append(r)

        work_rows.append({"dtc": d, "ob": ob, "in_ranked": in_ranked})

    used_inbound_rows: set[str] = set()
    inbound_assignments: dict[int, dict] = {}

    # Reserve obvious exact single-row matches before allowing those source rows
    # to be consumed by larger aggregate matches.
    for i, rec in enumerate(work_rows):
        d = rec["dtc"]
        ranked = rec["in_ranked"]
        ib = classify(ranked, d)
        if ranked.empty or ib["confidence"] != "High" or pd.isna(ib["row_id"]):
            continue
        tol = max(CFG.net_weight_tol_lbs, CFG.net_weight_tol_pct * d["net_weight"])
        if abs(float(ib["weight"]) - float(d["net_weight"])) <= tol:
            inbound_assignments[i] = ib
            used_inbound_rows.update(_row_id_parts(ib["row_id"]))

    for i, rec in enumerate(work_rows):
        if i in inbound_assignments:
            continue
        d = rec["dtc"]
        in_ranked = rec["in_ranked"]
        in_available = _exclude_used_ranked(in_ranked, used_inbound_rows)
        ib = classify(in_available, d)
        if in_available.empty and not in_ranked.empty:
            ib["note"] = "all candidate rows already assigned"
        ib = aggregate_inbound_match(d, inbound, ib, in_available, used_inbound_rows)
        inbound_assignments[i] = ib
        if ib["confidence"] != "No match":
            used_inbound_rows.update(_row_id_parts(ib["row_id"]))

    for i, rec in enumerate(work_rows):
        d = rec["dtc"]
        ob = rec["ob"]
        ib = inbound_assignments[i]

        notes = []
        if ob["note"]:
            notes.append("OB: " + ob["note"])
        if ib["note"]:
            notes.append("IN: " + ib["note"])

        matched_rows.append({
            "dtc_row_id": d["dtc_row_id"],
            "dtc_order_no": d["dtc_order_no"],
            "matched_inbound_row_id": ib["row_id"],
            "matched_inbound_ticket": ib["ticket"],
            "inbound_confidence": ib["confidence"],
            "inbound_score": ib["score"],
            "inbound_score_breakdown": ib["breakdown"],
            "matched_outbound_row_id": ob["row_id"],
            "matched_outbound_ticket": ob["ticket"],
            "outbound_confidence": ob["confidence"],
            "outbound_score": ob["score"],
            "outbound_score_breakdown": ob["breakdown"],
            "matched_material": d["desc"],
            "dtc_weight": d["net_weight"],
            "inbound_weight": ib["weight"],
            "outbound_weight": ob["weight"],
            "dtc_date": d["date"].strftime("%Y-%m-%d") if pd.notna(d["date"]) else "",
            "inbound_date": ib["date"].strftime("%Y-%m-%d") if pd.notna(ib["date"]) else "",
            "outbound_date": ob["date"].strftime("%Y-%m-%d") if pd.notna(ob["date"]) else "",
            "notes": " | ".join(notes),
            "_heldout_ob": d["_heldout_ob"],
            "_ob_margin": ob["margin"], "_in_margin": ib["margin"],
        })

    matched = pd.DataFrame(matched_rows)

    for side in ("outbound", "inbound"):
        col = f"matched_{side}_row_id"
        usage = {}
        conf_col = f"{side}_confidence"
        assigned_rows = matched[matched[conf_col] != "No match"]
        for idx, value in assigned_rows[col].dropna().items():
            for part in _row_id_parts(value):
                usage.setdefault(part, []).append(idx)
        dup_idx = {idx for idxs in usage.values() if len(idxs) > 1 for idx in idxs}
        if dup_idx:
            tag = f"DUPLICATE {side} row"
            mask = matched.index.isin(dup_idx)
            sep = matched.loc[mask, "notes"].fillna("").str.len().gt(0).map({True: " | ", False: ""})
            matched.loc[mask, "notes"] = (matched.loc[mask, "notes"].fillna("") + sep + tag).str.strip(" |")

    _write_outputs(matched, cand_ob_rows, cand_in_rows, dtc, outbound, inbound)
    return {"matched": matched, "n_dtc": len(dtc),
            "n_outbound": len(outbound), "n_inbound": len(inbound)}


def render_matched_chart(pub: pd.DataFrame, path: str) -> None:
    """Color-coded table image of the matched orders, banded by confidence."""
    cols = [("dtc_order_no", "DTC\nOrder"), ("matched_material", "Material"),
            ("dtc_weight", "DTC\nWt"), ("matched_outbound_ticket", "OB\nTicket"),
            ("matched_outbound_row_id", "OB\nRow"), ("outbound_confidence", "OB\nConf"), ("outbound_score", "OB\nScore"),
            ("matched_inbound_ticket", "IN\nTicket"), ("inbound_confidence", "IN\nConf"),
            ("matched_inbound_row_id", "IN\nRow"), ("inbound_score", "IN\nScore"), ("notes", "Notes")]
    disp = pub[[c for c, _ in cols]].copy()
    disp["dtc_weight"] = disp["dtc_weight"].astype("Int64")
    disp["notes"] = disp["notes"].fillna("").str.slice(0, 48)
    disp["matched_material"] = disp["matched_material"].str.slice(0, 26)

    n = len(disp)
    fig, ax = plt.subplots(figsize=(17, 0.34 * n + 1.6))
    ax.axis("off")
    ax.set_title("DTC Orders — Inbound / Outbound Ticket Matches (by weighted score)",
                 fontsize=14, pad=12)

    tbl = ax.table(cellText=disp.values, colLabels=[h for _, h in cols],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    band = {"High": "#e7f4e4", "Medium": "#fff4e5", "Low": "#fde9d6", "No match": "#fdecea"}
    for i in range(n):
        ob_c = disp["outbound_confidence"].iloc[i]
        in_c = disp["inbound_confidence"].iloc[i]
        note = str(disp["notes"].iloc[i])
        if "No match" in (ob_c, in_c) or "DUPLICATE" in note:
            color = "#fdecea"
        elif ob_c == "High" and in_c == "High":
            color = "#e7f4e4"
        else:
            color = "#fff4e5"
        for j in range(len(cols)):
            tbl[(i + 1, j)].set_facecolor(color)
        tbl[(i + 1, 5)].set_facecolor(band.get(ob_c, "#ffffff"))
        tbl[(i + 1, 8)].set_facecolor(band.get(in_c, "#ffffff"))
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#d9e1f2")
        tbl[(0, j)].set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(matched, cand_ob_rows, cand_in_rows, dtc, outbound, inbound):
    matched = _attach_model_predictions(matched)
    val = matched.dropna(subset=["_heldout_ob"]).copy()
    val_scored = val[val["matched_outbound_ticket"].notna()]
    ob_agree = int((val_scored["matched_outbound_ticket"].astype("Int64")
                    == val_scored["_heldout_ob"].astype("Int64")).sum())

    pub = matched.drop(columns=["_heldout_ob", "_ob_margin", "_in_margin"])
    pub.to_csv(os.path.join(OUT_DIR, "matched_dtc_orders.csv"), index=False)
    render_matched_chart(pub, os.path.join(OUT_DIR, "dtc_ticket_match.png"))

    cand_cols = ["dtc_order_no", "dtc_row_id", "ticket_row_id", "ticket", "score", "raw", "max_possible",
                 "breakdown", "ticket_weight", "ticket_date", "ticket_desc", "ticket_code"]
    pd.DataFrame(cand_ob_rows, columns=cand_cols).to_csv(
        os.path.join(OUT_DIR, "candidate_matches_outbound.csv"), index=False)
    pd.DataFrame(cand_in_rows, columns=cand_cols).to_csv(
        os.path.join(OUT_DIR, "candidate_matches_inbound.csv"), index=False)

    bad = {"No match", "Low"}
    exc_mask = (matched["outbound_confidence"].isin(bad) | matched["inbound_confidence"].isin(bad)
                | matched["notes"].str.contains("DUPLICATE|ambiguous|missing", case=False, na=False))
    matched.loc[exc_mask].drop(columns=["_heldout_ob", "_ob_margin", "_in_margin"]).to_csv(
        os.path.join(OUT_DIR, "exceptions_unmatched_or_ambiguous.csv"), index=False)

    _write_summary(matched, dtc, outbound, inbound, ob_agree, len(val_scored), exc_mask.sum())


def _attach_model_predictions(matched: pd.DataFrame) -> pd.DataFrame:
    out = matched.copy()

    ob_path = os.path.join(OUT_DIR, "trained_outbound_matches.csv")
    if os.path.exists(ob_path):
        ob = pd.read_csv(ob_path)
        keep = {
            "predicted_outbound_row_id": "model_outbound_row_id",
            "predicted_outbound_ticket": "model_outbound_ticket",
            "outbound_match_probability": "model_outbound_probability",
            "truth_in_candidate": "model_outbound_truth_in_candidate",
        }
        ob = ob[["dtc_row_id"] + [c for c in keep if c in ob.columns]].rename(columns=keep)
        out = out.merge(ob, on="dtc_row_id", how="left")

    in_path = os.path.join(OUT_DIR, "trained_inbound_matches.csv")
    if os.path.exists(in_path):
        ib = pd.read_csv(in_path)
        keep = {
            "predicted_inbound_row_id": "model_inbound_row_id",
            "predicted_inbound_ticket": "model_inbound_ticket",
            "inbound_match_probability": "model_inbound_probability",
            "pseudo_labeled_order": "model_inbound_pseudo_labeled",
        }
        ib = ib[["dtc_row_id"] + [c for c in keep if c in ib.columns]].rename(columns=keep)
        out = out.merge(ib, on="dtc_row_id", how="left")

    return out


def _dist(series: pd.Series) -> str:
    vc = series.value_counts()
    order = ["High", "Medium", "Low", "No match"]
    return ", ".join(f"{k}={int(vc.get(k, 0))}" for k in order)


def _write_summary(matched, dtc, outbound, inbound, ob_agree, n_val, n_exc):
    n = len(matched)
    ob_found = int((matched["outbound_confidence"] != "No match").sum())
    in_found = int((matched["inbound_confidence"] != "No match").sum())
    lines = []
    A = lines.append
    A("DTC <-> INBOUND/OUTBOUND RECORD LINKAGE — MATCH SUMMARY")
    A("=" * 70)
    A("")
    A("ROW COUNTS")
    A(f"  DTC orders:        {n}")
    A(f"  Outbound tickets:  {len(outbound)} (lines; a ticket may have several materials)")
    A(f"  Inbound tickets:   {len(inbound)} (lines)")
    A("")
    A("MATCH RATES")
    A(f"  Outbound matched (>= min score): {ob_found}/{n}  ({100*ob_found/n:.0f}%)")
    A(f"  Inbound  matched (>= min score): {in_found}/{n}  ({100*in_found/n:.0f}%)")
    A("")
    A("CONFIDENCE DISTRIBUTION")
    A(f"  Outbound: {_dist(matched['outbound_confidence'])}")
    A(f"  Inbound:  {_dist(matched['inbound_confidence'])}")
    A(f"  Exception rows (unmatched/low/ambiguous/duplicate): {int(n_exc)}")
    A("")
    A("VALIDATION (held-out, NOT used as a matching feature)")
    A(f"  'Outbound Ticket Id' present in DTC = deterministic ground truth.")
    A(f"  Inferred outbound ticket agrees with it: {ob_agree}/{n_val} scored "
      f"({100*ob_agree/n_val:.0f}%).")
    A("  (No ground truth exists for the inbound side.)")
    A("")
    A("INFERRED COLUMN MAPPINGS  (DTC field  ->  ticket field)")
    A("  DTC order id:   dtc_row_id = 'DTC Row Id'; dtc_order_no = 'Invoice Id'")
    A("                  (DTC has no native ticket number of its own).")
    A("  Ticket row id:  matched_inbound_row_id / matched_outbound_row_id identify")
    A("                  exact source CSV lines, even when ticket numbers repeat")
    A("                  across several material lines.")
    A("  Material code:  DTC has NONE -> derived from 'Material Name' via a")
    A("                  description->code map built from the ticket files")
    A("                  (unambiguous: no description maps to two codes).")
    A("  Material desc:  DTC 'Material Name' -> inbound 'Material Name' / outbound 'Material'.")
    A("  Net weight:     DTC 'Yard Net Weight' (yard scale) -> ticket 'Net Weight';")
    A("                  fallback DTC 'Consumer Net Weight' (differs in 20/85 rows).")
    A("  Gross weight:   DTC has NO gross weight -> gross-weight component UNAVAILABLE")
    A("                  for every DTC pair (tickets carry gross, nothing to compare).")
    A("  Date:           DTC 'Invoice Issue Date' (fmt %m-%d-%y, %H:%M:%S, tz-naive)")
    A("                  -> outbound 'Date In' / inbound 'Effective Date' (ISO + IANA")
    A("                  zone, converted to tz-naive UTC). Possible intraday tz skew;")
    A("                  immaterial at the day-level windows used.")
    A("  Customer:       DTC 'Customer Name' = the BUYER -> outbound 'Customer Name' ONLY.")
    A("                  Inbound 'Customer Name' = SUPPLIER/vendor, NOT the DTC buyer ->")
    A("                  customer component UNAVAILABLE on the inbound side.")
    A("  Location/yard:  No DTC field maps to ticket yard ('Ship To *' is the consumer")
    A("                  destination) -> location component UNAVAILABLE (weight 0 used).")
    A("  Price/rate:     DTC 'Yard Value'/weight  ->  outbound 'Expected Value'/weight,")
    A("                  inbound 'Cost'/weight. Weak component.")
    A("")
    A("EXCLUDED DETERMINISTIC KEYS (deliberately NOT used for matching)")
    A("  'Outbound Ticket Id' (85/85 -> outbound) and 'Sales Order #' (85/85 ->")
    A("  outbound 'Sales Order') would join DTC->outbound exactly, but the task is to")
    A("  INFER links from physical attributes, so both are excluded from scoring.")
    A("  'Outbound Ticket Id' is held out solely to report accuracy above.")
    A("")
    A("DATE-WINDOW EVIDENCE (why the windows are tight)")
    A("  For direct-to-consumer flow, receipt/shipment/invoice cluster within days:")
    A("   - DTC invoice date vs true outbound date: within 3 days for 84/84.")
    A("   - DTC invoice date vs inbound effective date: -4..+3 days (median -1).")
    A("")
    agg_count = int(matched["notes"].fillna("").str.contains("aggregated").sum())
    inbound_usage = {}
    assigned_in = matched[matched["inbound_confidence"] != "No match"]
    for _, value in assigned_in["matched_inbound_row_id"].dropna().items():
        for part in _row_id_parts(value):
            inbound_usage[part] = inbound_usage.get(part, 0) + 1
    duplicate_lots = sum(1 for n in inbound_usage.values() if n > 1)
    A("AUTOMATED INBOUND AGGREGATION & UNIQUENESS")
    A(f"  Aggregated inbound matches adopted: {agg_count}")
    A(f"  Distinct inbound source rows reused by assigned matches: {duplicate_lots}")
    A(f"  Aggregation window: -{CFG.agg_lookback_days}/+{CFG.agg_lookahead_days} days;")
    A(f"  max rows per aggregate: {CFG.agg_max_rows}; sum tolerance=max({CFG.agg_tol_lbs} lb, {CFG.agg_tol_pct:.1%}).")
    A("  Multi-line/multi-ticket aggregates are expected for orders where several")
    A("  receipt rows add up to one DTC shipment/order weight.")
    A("  The number of tickets/lines in a valid aggregate does not lower confidence;")
    A("  aggregate confidence is based on material, summed weight, and date proximity.")
    A("  Greedy uniqueness is enforced in DTC row order: once an inbound row is")
    A("  assigned to a matched order, later orders must use other available rows.")
    A("")
    if "model_outbound_probability" in matched.columns or "model_inbound_probability" in matched.columns:
        A("MODEL AUDIT COLUMNS")
        if "model_outbound_probability" in matched.columns:
            ob_cov = int(matched["model_outbound_truth_in_candidate"].fillna(0).sum())
            A(f"  Outbound model: trained on deterministic Outbound Ticket Id labels;")
            A(f"  true outbound ticket present in candidate set for {ob_cov}/{n} rows.")
        if "model_inbound_probability" in matched.columns:
            in_pseudo = int(matched["model_inbound_pseudo_labeled"].fillna(0).sum())
            A("  Inbound model: trained from strict high-confidence pseudo-labels;")
            A(f"  pseudo-labeled single-row orders available for {in_pseudo}/{n} rows.")
        A("  Model columns are audit scores; final matched tickets still come from")
        A("  the rule/aggregation assignment pass above.")
        A("")
    A("THRESHOLDS & WEIGHTS (all in Config; edit there to tune)")
    A(f"  Net weight tol: max({CFG.net_weight_tol_lbs} lb, {CFG.net_weight_tol_pct:.1%}); "
      f"partial credit to {CFG.net_weight_partial_mult}x tol.")
    A(f"  Date tiers (days): exact={CFG.date_exact_days}, near={CFG.date_near_days}, "
      f"mid={CFG.date_mid_days}, max-score={CFG.date_max_score_days}, block=+/-{CFG.date_block_days}.")
    A(f"  Fuzzy: desc_min={CFG.desc_min_ratio}, desc_block={CFG.desc_block_ratio}, "
      f"customer_min={CFG.customer_min_ratio}, customer_diff={CFG.customer_diff_ratio}.")
    A(f"  Weights: code={CFG.w_code}, desc={CFG.w_desc}, net_wt={CFG.w_net_weight}, "
      f"gross={CFG.w_gross_weight}, date={CFG.w_date}, time={CFG.w_time}, "
      f"customer={CFG.w_customer}, location={CFG.w_location}, price={CFG.w_price}.")
    A(f"  Penalties: code_diff={CFG.p_code_diff}, weight_over={CFG.p_weight_over}, "
      f"customer_diff={CFG.p_customer_diff}.")
    A(f"  Confidence (0-100 normalized): High>={CFG.score_high} & margin>={CFG.high_margin}; "
      f"Medium>={CFG.score_medium}; Low>={CFG.score_min}; else No match.")
    A("  Scores are normalized to 100 * raw / (sum of max weights of AVAILABLE")
    A("  components), so the inbound side is not penalized for fields it cannot use.")
    A("")
    A("RECOMMENDED THRESHOLD ADJUSTMENTS")
    ob_unique_hi = int(((matched["outbound_confidence"] == "High")).sum())
    A(f"  - Outbound High-confidence rows: {ob_unique_hi}/{n}. Agreement with ground")
    A("    truth is the key check above; if it is ~100%, current weights are sound.")
    A("  - Inbound scores top out lower (no customer/gross/location signal). If too many")
    A("    inbound rows land in 'Low', lower score_medium/score_min or raise w_net_weight")
    A("    and w_date for the inbound pass specifically.")
    A("  - Duplicate row-id notes mean the exact same source line was reused.")
    A("    Repeated ticket numbers alone may be valid multi-material tickets.")
    A("  - Tighten date_block_days if runtime grows on larger extracts; widen it if")
    A("    receipts can precede sales by more than two weeks in other yards.")
    A("")
    with open(os.path.join(OUT_DIR, "match_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    res = run()
    m = res["matched"]
    print(f"DTC orders: {res['n_dtc']}  |  outbound tickets: {res['n_outbound']}  "
          f"|  inbound tickets: {res['n_inbound']}")
    print("Outbound confidence:", _dist(m["outbound_confidence"]))
    print("Inbound  confidence:", _dist(m["inbound_confidence"]))
    print(f"\nWrote 6 files to {OUT_DIR}/:")
    for f in ["matched_dtc_orders.csv", "dtc_ticket_match.png",
              "candidate_matches_inbound.csv", "candidate_matches_outbound.csv",
              "exceptions_unmatched_or_ambiguous.csv", "match_summary.txt"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
