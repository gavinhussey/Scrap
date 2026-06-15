"""Match each DTC order to its partner inbound and outbound tickets by weight."""

import os
from itertools import combinations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WINDOW_DAYS = 1
MAX_COMBO = 6

CODE_EQUIV = {"FETURN": {"FETURN", "TURNSTEEL"}, "TURNSTEEL": {"FETURN", "TURNSTEEL"}}


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce")


def _subset(items, target):
    """Return the 'best' subset (list of (ticket,wt,date)) summing to target, or None.
    Prefers fewest tickets, then the tightest date span (most likely a real bin)."""
    target = int(round(target))
    pool = [(t, int(round(w)), d) for t, w, d in items if 0 < round(w) <= target]

    best = None
    for r in range(2, min(MAX_COMBO, len(pool)) + 1):
        for combo in combinations(pool, r):
            if sum(w for _, w, _ in combo) == target:
                span = (max(d for *_, d in combo) - min(d for *_, d in combo)).days
                key = (r, span)
                if best is None or key < best[0]:
                    best = (key, combo)
        if best:
            break
    return list(best[1]) if best else None


def main():
    dtc = pd.read_csv(os.path.join(HERE, "Copy of MASTER ____ BMR-MMR Dashboard v11 - DTC_raw_data.csv"), dtype=str)
    inb = pd.read_csv(os.path.join(ROOT, "data", "2026 ytd combined inbound.csv"), dtype=str)
    out = pd.read_csv(os.path.join(ROOT, "data", "2026 ytd combined outbound.csv"), dtype=str)

    dtc["net"] = _num(dtc["Yard Net Weight"])
    inb["net"] = _num(inb["Net Weight"]); inb["gross"] = _num(inb["Gross Weight"])
    out["net"] = _num(out["Net Weight"]); out["gross"] = _num(out["Gross Weight"])
    inb["date"] = pd.to_datetime(inb["Effective Date"].str[:10], errors="coerce")
    out["date"] = pd.to_datetime(out["Date In"].str[:10], errors="coerce")

    itix = inb.groupby(["Material Code", "Location", "Ticket #"], as_index=False).agg(
        net=("net", "sum"), gross=("gross", "max"), date=("date", "min"),
        supplier=("Customer Name", "first"), vendor_class=("Vendor Class", "first"))
    itix["tid"] = itix["Location"] + "/" + itix["Ticket #"]
    sup = itix.set_index("tid")[["gross", "supplier", "vendor_class"]]

    recs = []
    for _, o in dtc.iterrows():
        W = o["net"]
        tkt = o["Outbound Ticket Id"]
        rec = dict(dtc_row=o["DTC Row Id"], outbound_ticket=tkt, material=o["Material Name"],
                   dtc_net=W, out_method="", out_match="", out_gross="",
                   mat_code="", yard="", ship_date="",
                   in_method="NONE", in_match="", in_total="", consumer=o["Customer Name"],
                   _W=W, _cand=None, _og=None, _intid=None)

        ot = out[out["Outbound Ticket #"] == tkt]
        hit = ot[ot["net"] == W]
        if len(hit):
            r = hit.iloc[0]
            rec.update(out_method="exact net (linked ticket)", out_match=tkt,
                       out_gross=r["gross"], mat_code=r["Material Code"],
                       yard=r["Location"], ship_date=str(r["date"].date()) if pd.notna(r["date"]) else "")
        elif len(ot):
            rec.update(out_method="ticket found, net mismatch", out_match=tkt,
                       mat_code=ot.iloc[0]["Material Code"], yard=ot.iloc[0]["Location"],
                       ship_date=str(ot.iloc[0]["date"].date()) if pd.notna(ot.iloc[0]["date"]) else "",
                       out_gross=ot["gross"].max())
        else:
            rec.update(out_method="ticket NOT in outbound file")

        mc, yd, sd = rec["mat_code"], rec["yard"], rec["ship_date"]
        if mc and yd and sd:
            ship = pd.Timestamp(sd)
            codes = CODE_EQUIV.get(mc, {mc})
            pool = itix[(itix["Material Code"].isin(codes)) & (itix["Location"] == yd)].copy()
            rec["_pool"] = pool
            rec["_cand"] = pool[(pool["date"] >= ship - pd.Timedelta(days=WINDOW_DAYS)) &
                                (pool["date"] <= ship + pd.Timedelta(days=WINDOW_DAYS))].copy()
            rec["_ship"] = ship
            rec["_og"] = pd.to_numeric(pd.Series([rec["out_gross"]]), errors="coerce").iloc[0]
        recs.append(rec)

    consumed = set()
    order = sorted(range(len(recs)),
                   key=lambda i: recs[i].get("_ship", pd.Timestamp.max))

    def avail(c):
        return c[~c["tid"].isin(consumed)] if c is not None else None

    for i in order:
        r = recs[i]; c = avail(r["_cand"])
        if c is None or not len(c):
            continue
        ex = c[c["net"].round() == round(r["_W"])]
        if len(ex):
            ex = ex.assign(_d=(ex["date"] - r["_ship"]).abs()).sort_values(["_d"])
            pick = ex.iloc[0]
            consumed.add(pick["tid"])
            r.update(in_method="exact net", in_match=f"#{pick['Ticket #']}", in_total=round(pick["net"]), _intid=pick["tid"])

    for i in order:
        r = recs[i]
        if r["in_method"] != "NONE":
            continue
        c = avail(r["_cand"])
        if c is None or not len(c):
            continue
        s = _subset(list(zip(c["tid"], c["net"], c["date"])), r["_W"])
        if s:
            for tid, *_ in s:
                consumed.add(tid)
            r.update(in_method="sum of nets",
                     in_match=" + ".join(f"#{tid.split('/')[1]}({int(round(w))})" for tid, w, _ in s),
                     in_total=sum(int(round(w)) for _, w, _ in s))

    for i in order:
        r = recs[i]
        if r["in_method"] != "NONE" or pd.isna(r["_og"]):
            continue
        c = avail(r["_cand"])
        if c is None or not len(c):
            continue
        exg = c[c["gross"].round() == round(r["_og"])]
        if len(exg):
            pick = exg.iloc[0]; consumed.add(pick["tid"])
            r.update(in_method="exact gross", in_match=f"#{pick['Ticket #']}", in_total=round(pick["gross"]), _intid=pick["tid"])
            continue
        s = _subset(list(zip(c["tid"], c["gross"], c["date"])), r["_og"])
        if s:
            for tid, *_ in s:
                consumed.add(tid)
            r.update(in_method="sum of gross",
                     in_match=" + ".join(f"#{tid.split('/')[1]}({int(round(w))})" for tid, w, _ in s),
                     in_total=sum(int(round(w)) for _, w, _ in s))

    for i in order:
        r = recs[i]
        if r["in_method"] != "NONE" or r.get("_pool") is None:
            continue
        p = avail(r["_pool"])
        if p is None or not len(p):
            continue

        ex = p[p["net"].round() == round(r["_W"])]
        if len(ex):
            ex = ex.assign(_d=(ex["date"] - r["_ship"]).abs()).sort_values("_d")
            pick = ex.iloc[0]; consumed.add(pick["tid"])
            gap = (pick["date"] - r["_ship"]).days
            r.update(in_method="exact net (outside same-day)", in_match=f"#{pick['Ticket #']}",
                     in_total=round(pick["net"]), _gap=gap, _intid=pick["tid"])

    for r in recs:
        r.update(in_gross="", gross_check="", supplier="", supplier_class="", supplier_check="", confidence="")
        tid = r.get("_intid")
        if tid is None or tid not in sup.index:
            continue
        row = sup.loc[tid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        ig, og = row["gross"], r["_og"]
        r["in_gross"] = int(round(ig)) if pd.notna(ig) else ""
        r["gross_check"] = ("net+gross" if pd.notna(ig) and pd.notna(og) and round(ig) == round(og)
                            else f"NET ONLY (gross {int(round(ig))} vs out {int(round(og))})" if pd.notna(ig) and pd.notna(og)
                            else "gross n/a")
        r["supplier"] = row["supplier"]
        r["supplier_class"] = row["vendor_class"]
        same = str(row["supplier"]).strip().lower() == str(r["consumer"]).strip().lower()
        r["supplier_check"] = "SUSPECT: supplier==consumer" if same else "ok (distinct counterparties)"

    no_twin = set(_no_inbound_twin_anywhere(dtc, inb, out))
    for r in recs:
        m = r["in_method"]
        gross_ok = r["gross_check"] == "net+gross"
        supp_ok = r["supplier_check"].startswith("ok")
        if "outside same-day" in m:
            r["confidence"] = "Medium" if gross_ok else "Low"
            r["exception_reason"] = f"inbound ticket(s) shown but {r.get('_gap', '?')}d outside same-day (not stored = should be same day)"
        elif m != "NONE":
            r["confidence"] = "High" if (gross_ok and supp_ok) else "Low"
            r["exception_reason"] = "" if r["confidence"] == "High" else (
                "net matches but GROSS differs (likely coincidental weight collision)" if not gross_ok
                else "supplier == consumer (implausible for a pass-through)")
        elif r["out_method"] == "ticket NOT in outbound file":
            r["exception_reason"] = "outbound ticket missing from outbound file"
        elif r["dtc_row"] in no_twin:
            r["exception_reason"] = "no inbound ticket of this net anywhere (missing inbound leg / not a pass-through)"
        else:
            r["exception_reason"] = "net twin exists only under a different material/yard (coincidental)"

    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    res.to_csv(os.path.join(HERE, "dtc_weight_matches.csv"), index=False)
    exc = res[res["exception_reason"] != ""].copy()
    exc.to_csv(os.path.join(HERE, "dtc_match_exceptions.csv"), index=False)

    n = len(res)
    print(f"DTC orders: {n}\n")
    print("OUTBOUND match:")
    print(res["out_method"].value_counts().to_string(), "\n")
    high = (res["confidence"] == "High").sum()
    matched = (res["in_method"] != "NONE").sum()
    print("CONFIDENCE (net+gross corroborated, supplier != consumer):")
    print(f"  High (net+gross+supplier ok)      : {high}")
    print(f"  matched but flagged for review    : {matched - high}")
    print(f"  no inbound found                  : {n - matched}")
    print("\nException reasons:")
    print(exc["exception_reason"].value_counts().to_string())
    print("\nExceptions to chase:")
    print(exc[["dtc_row", "outbound_ticket", "material", "dtc_net", "yard", "ship_date", "exception_reason"]].to_string(index=False))
    render_chart(res, os.path.join(HERE, "dtc_ticket_match.png"))
    print(f"\n-> wrote dtc/dtc_weight_matches.csv, dtc/dtc_match_exceptions.csv, dtc/dtc_ticket_match.png")


def render_chart(res, path):
    """Color-coded table image: green = High (net+gross+supplier), amber = flagged, red = none."""
    cols = [("dtc_row", "DTC\nRow"), ("material", "Material"), ("dtc_net", "DTC\nNet"),
            ("outbound_ticket", "OB\nTicket"), ("yard", "Yard"), ("ship_date", "Ship\nDate"),
            ("in_match", "Inbound\nTicket(s)"), ("gross_check", "Gross\nCheck"),
            ("supplier", "Inbound Supplier"), ("confidence", "Conf"),
            ("exception_reason", "Exception / Note")]
    disp = res[[c for c, _ in cols]].copy()
    disp["material"] = disp["material"].str.slice(0, 22)
    disp["supplier"] = disp["supplier"].fillna("").str.slice(0, 22)
    disp["gross_check"] = disp["gross_check"].fillna("").str.slice(0, 20)
    disp["exception_reason"] = disp["exception_reason"].fillna("")
    disp["in_match"] = disp["in_match"].fillna("").str.slice(0, 22)

    n = len(disp)
    fig, ax = plt.subplots(figsize=(24, 0.32 * n + 1.8))
    ax.axis("off")
    high = int((res["confidence"] == "High").sum())
    matched = int((res["in_method"] != "NONE").sum())
    ax.set_title(f"DTC Orders — Inbound/Outbound Matches, net+gross+supplier corroborated  "
                 f"(dark-green {high} High | light-green {matched - high} matched/flagged | red {n - matched} no inbound)",
                 fontsize=13, pad=12)

    tbl = ax.table(cellText=disp.values, colLabels=[h for _, h in cols],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    for i in range(n):
        if res["in_method"].iloc[i] == "NONE":
            color = "#fdecea"
        elif res["confidence"].iloc[i] == "High":
            color = "#a8dba0"
        else:
            color = "#d7f0cf"
        for j in range(len(cols)):
            tbl[(i + 1, j)].set_facecolor(color)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#d9e1f2")
        tbl[(0, j)].set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _no_inbound_twin_anywhere(dtc, inb, out):
    """DTC rows whose net equals no single inbound ticket net on ANY date, either yard."""
    itix = inb.groupby(["Location", "Ticket #"], as_index=False)["net"].sum()
    nets = set(itix["net"].round())
    return [o["DTC Row Id"] for _, o in dtc.iterrows() if round(o["net"]) not in nets]


if __name__ == "__main__":
    main()
