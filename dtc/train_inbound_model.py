"""Train an inbound DTC ticket ranker from strict pseudo-labels.

Inbound has no deterministic ground-truth key equivalent to DTC
"Outbound Ticket Id". This script therefore uses weak supervision:
only very clean top-ranked inbound candidates are treated as pseudo-positive
training examples. The model scores all candidates, but evaluation metrics
are against pseudo-labels, not independent truth.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dtc_ticket_match as matcher


DTC_PATH = os.path.join(HERE, "Copy of MASTER ____ BMR-MMR Dashboard v11 - DTC_raw_data.csv")
FULL_CANDIDATES_PATH = os.path.join(HERE, "inbound_training_candidates_full.csv")
FEATURES_PATH = os.path.join(HERE, "inbound_training_features.csv")
SCORED_CANDIDATES_PATH = os.path.join(HERE, "trained_inbound_scored_candidates.csv")
MATCHES_PATH = os.path.join(HERE, "trained_inbound_matches.csv")
PSEUDO_LABELS_PATH = os.path.join(HERE, "inbound_pseudo_labels.csv")
EVAL_PATH = os.path.join(HERE, "inbound_model_evaluation.txt")
MODEL_PATH = os.path.join(HERE, "inbound_match_model.json")

DATE_FMT = "%m-%d-%y, %H:%M:%S"
MONEY_RE = re.compile(r"[,$\s]")

PSEUDO_MIN_SCORE = 85.0
PSEUDO_MIN_MARGIN = 10.0
PSEUDO_MAX_DATE_ABS_DAYS = 1
PSEUDO_WEIGHT_TOL_LBS = 1.0
PSEUDO_WEIGHT_TOL_PCT = 0.005

FEATURE_COLUMNS = [
    "rule_score",
    "rule_raw",
    "rule_max_possible",
    "rule_rank",
    "score_gap_to_best",
    "dtc_weight",
    "ticket_weight",
    "weight_abs_diff",
    "weight_pct_diff",
    "date_abs_days",
    "date_signed_days",
    "same_day",
    "code_points",
    "code_diff_points",
    "desc_points",
    "net_wt_points",
    "net_wt_over_points",
    "date_points",
    "time_points",
    "price_points",
]


@dataclass
class FittedModel:
    feature_columns: list[str]
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: float


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce")


def _money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(MONEY_RE, "", regex=True), errors="coerce")


def _breakdown_value(text: object, key: str) -> float:
    if pd.isna(text):
        return 0.0
    for part in str(text).split(";"):
        if part.startswith(key + "+") or part.startswith(key):
            try:
                return float(part[len(key):])
            except ValueError:
                return 0.0
    return 0.0


def build_full_candidate_set() -> pd.DataFrame:
    code_map = matcher._build_code_map()
    dtc = matcher.load_dtc(code_map)
    inbound = matcher.load_inbound()

    rows = []
    for _, d in dtc.iterrows():
        ranked = matcher.rank_candidates(d, inbound, "inbound")
        if ranked.empty:
            continue
        ranked = ranked.copy()
        ranked["dtc_order_no"] = d["dtc_order_no"]
        rows.append(ranked)

    if not rows:
        raise RuntimeError("No inbound candidates generated")

    out = pd.concat(rows, ignore_index=True)
    cols = [
        "dtc_order_no", "dtc_row_id", "ticket_row_id", "ticket", "score", "raw",
        "max_possible", "breakdown", "ticket_weight", "ticket_date", "ticket_desc",
        "ticket_code",
    ]
    return out[cols].sort_values(["dtc_row_id", "score"], ascending=[True, False])


def build_features() -> pd.DataFrame:
    dtc = pd.read_csv(DTC_PATH, thousands=",")
    cand = build_full_candidate_set()
    cand.to_csv(FULL_CANDIDATES_PATH, index=False)

    dtc_features = pd.DataFrame({
        "dtc_row_id": pd.to_numeric(dtc["DTC Row Id"], errors="coerce").astype("Int64"),
        "dtc_weight": _num(dtc["Yard Net Weight"]).fillna(_num(dtc["Consumer Net Weight"])),
        "dtc_date": pd.to_datetime(dtc["Invoice Issue Date"], format=DATE_FMT, errors="coerce"),
        "dtc_value": _money(dtc["Yard Value"]),
    })

    out = cand.merge(dtc_features, on="dtc_row_id", how="left")
    out["ticket"] = pd.to_numeric(out["ticket"], errors="coerce").astype("Int64")
    out["rule_score"] = pd.to_numeric(out["score"], errors="coerce")
    out["rule_raw"] = pd.to_numeric(out["raw"], errors="coerce")
    out["rule_max_possible"] = pd.to_numeric(out["max_possible"], errors="coerce")
    out["ticket_weight"] = pd.to_numeric(out["ticket_weight"], errors="coerce")
    out["ticket_date"] = pd.to_datetime(out["ticket_date"], errors="coerce")

    out["rule_rank"] = out.groupby("dtc_row_id")["rule_score"].rank(method="first", ascending=False)
    top_score = out.groupby("dtc_row_id")["rule_score"].transform("max")
    out["score_gap_to_best"] = top_score - out["rule_score"]
    second = out[out["rule_rank"].eq(2)][["dtc_row_id", "rule_score"]].rename(
        columns={"rule_score": "second_rule_score"}
    )
    out = out.merge(second, on="dtc_row_id", how="left")
    out["pseudo_margin"] = top_score - out["second_rule_score"].fillna(0.0)

    out["weight_abs_diff"] = (out["dtc_weight"] - out["ticket_weight"]).abs()
    out["weight_pct_diff"] = out["weight_abs_diff"] / out["dtc_weight"].replace(0, np.nan)
    out["pseudo_weight_tol"] = np.maximum(
        PSEUDO_WEIGHT_TOL_LBS, PSEUDO_WEIGHT_TOL_PCT * out["dtc_weight"]
    )
    date_delta = out["ticket_date"].dt.normalize() - out["dtc_date"].dt.normalize()
    out["date_signed_days"] = date_delta.dt.days
    out["date_abs_days"] = out["date_signed_days"].abs()
    out["same_day"] = (out["date_abs_days"] == 0).astype(float)

    breakdown_keys = {
        "code_points": "code",
        "code_diff_points": "code_diff",
        "desc_points": "desc",
        "net_wt_points": "net_wt",
        "net_wt_over_points": "net_wt_over",
        "date_points": "date",
        "time_points": "time",
        "price_points": "price",
    }
    for col, key in breakdown_keys.items():
        out[col] = out["breakdown"].map(lambda text, k=key: _breakdown_value(text, k))

    top_clean = (
        out["rule_rank"].eq(1)
        & out["rule_score"].ge(PSEUDO_MIN_SCORE)
        & out["pseudo_margin"].ge(PSEUDO_MIN_MARGIN)
        & out["weight_abs_diff"].le(out["pseudo_weight_tol"])
        & out["date_abs_days"].le(PSEUDO_MAX_DATE_ABS_DAYS)
    )
    pseudo_positive_ids = set(out.loc[top_clean, "dtc_row_id"].astype(int))
    out["pseudo_labeled_order"] = out["dtc_row_id"].astype(int).isin(pseudo_positive_ids).astype(int)
    out["pseudo_label"] = (top_clean & out["pseudo_labeled_order"].eq(1)).astype(int)
    out["pseudo_label_reason"] = np.where(
        out["pseudo_label"].eq(1),
        "top_clean_score_margin_weight_date",
        "",
    )

    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    cols = [
        "dtc_row_id", "dtc_order_no", "ticket_row_id", "ticket", "pseudo_label",
        "pseudo_labeled_order", "pseudo_label_reason", "pseudo_margin",
    ] + FEATURE_COLUMNS
    return out[cols].sort_values(["dtc_row_id", "rule_rank"]).reset_index(drop=True)


def _standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-9] = 1.0
    return (x - means) / scales, means, scales


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(features: pd.DataFrame, *, l2: float = 0.03, lr: float = 0.08,
                   epochs: int = 6000) -> FittedModel:
    features = features[features["pseudo_labeled_order"].eq(1)].copy()
    if features.empty:
        raise RuntimeError("No pseudo-labeled inbound orders available for training")

    x = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = features["pseudo_label"].to_numpy(dtype=float)
    xz, means, scales = _standardize_train(x)

    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y == 1.0, n_neg / n_pos, 1.0)

    weights = np.zeros(xz.shape[1], dtype=float)
    bias = 0.0
    denom = sample_weight.sum()

    for _ in range(epochs):
        pred = _sigmoid(xz @ weights + bias)
        err = (pred - y) * sample_weight
        grad_w = (xz.T @ err) / denom + l2 * weights
        grad_b = err.sum() / denom
        weights -= lr * grad_w
        bias -= lr * grad_b

    return FittedModel(FEATURE_COLUMNS, means, scales, weights, float(bias))


def predict_proba(model: FittedModel, features: pd.DataFrame) -> np.ndarray:
    x = features[model.feature_columns].to_numpy(dtype=float)
    xz = (x - model.means) / model.scales
    return _sigmoid(xz @ model.weights + model.bias)


def _top_by(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (df.sort_values(["dtc_row_id", score_col], ascending=[True, False])
              .drop_duplicates("dtc_row_id", keep="first")
              .sort_values("dtc_row_id"))


def evaluate_predictions(scored: pd.DataFrame, score_col: str) -> dict[str, float]:
    eligible = scored[scored["pseudo_labeled_order"].eq(1)].copy()
    top = _top_by(eligible, score_col)
    top["correct"] = top["pseudo_label"].astype(int)
    return {
        "orders": float(top["dtc_row_id"].nunique()),
        "top1_correct": float(top["correct"].sum()),
        "top1_accuracy": float(top["correct"].mean()),
    }


def cross_validated_scores(features: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    rows = []
    eligible = features[features["pseudo_labeled_order"].eq(1)].copy()
    for fold in range(folds):
        test_ids = sorted(eligible.loc[eligible["dtc_row_id"] % folds == fold, "dtc_row_id"].unique())
        train = eligible[~eligible["dtc_row_id"].isin(test_ids)].copy()
        test = eligible[eligible["dtc_row_id"].isin(test_ids)].copy()
        if test.empty:
            continue
        model = train_logistic(train)
        test["model_probability"] = predict_proba(model, test)
        test["cv_fold"] = fold
        rows.append(test)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["dtc_row_id", "model_probability"], ascending=[True, False]
    )


def save_model(model: FittedModel, metrics: dict[str, object]) -> None:
    payload = {
        "model_type": "standardized_logistic_regression",
        "label_source": "strict inbound pseudo-labels from current rule scorer",
        "pseudo_label_rule": {
            "top_rank": 1,
            "min_score": PSEUDO_MIN_SCORE,
            "min_margin": PSEUDO_MIN_MARGIN,
            "max_date_abs_days": PSEUDO_MAX_DATE_ABS_DAYS,
            "weight_tolerance": f"max({PSEUDO_WEIGHT_TOL_LBS}, {PSEUDO_WEIGHT_TOL_PCT} * dtc_weight)",
        },
        "feature_columns": model.feature_columns,
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
        "weights": model.weights.tolist(),
        "bias": model.bias,
        "metrics": metrics,
    }
    with open(MODEL_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def write_evaluation(cv_scored: pd.DataFrame, final_scored: pd.DataFrame) -> dict[str, object]:
    baseline = evaluate_predictions(cv_scored, "rule_score")
    cv_model = evaluate_predictions(cv_scored, "model_probability")
    final_model = evaluate_predictions(final_scored, "model_probability")

    n_orders = int(final_scored["dtc_row_id"].nunique())
    n_pseudo_orders = int(final_scored.groupby("dtc_row_id")["pseudo_labeled_order"].max().sum())
    n_pseudo_pos = int(final_scored["pseudo_label"].sum())

    lines = []
    a = lines.append
    a("INBOUND MATCH MODEL EVALUATION")
    a("=" * 60)
    a("")
    a("Training label: strict pseudo-labels from high-confidence single-row inbound matches.")
    a("No deterministic inbound ground-truth key exists in the current data.")
    a("")
    a("Pseudo-label rule:")
    a(f"  top candidate, score >= {PSEUDO_MIN_SCORE}, margin >= {PSEUDO_MIN_MARGIN},")
    a(f"  date abs days <= {PSEUDO_MAX_DATE_ABS_DAYS},")
    a(f"  weight diff <= max({PSEUDO_WEIGHT_TOL_LBS}, {PSEUDO_WEIGHT_TOL_PCT:.3%} * DTC weight)")
    a("")
    a(f"Candidate rows: {len(final_scored)}")
    a(f"DTC orders: {n_orders}")
    a(f"Pseudo-labeled orders used for training/evaluation: {n_pseudo_orders}/{n_orders}")
    a(f"Pseudo-positive candidate rows: {n_pseudo_pos}")
    a("")
    a("GROUPED 5-FOLD CROSS-VALIDATION BY DTC ROW")
    a("  Evaluated only against pseudo-labeled orders.")
    a(f"  Rule-score top-1 accuracy vs pseudo-labels: "
      f"{baseline['top1_correct']:.0f}/{baseline['orders']:.0f} ({baseline['top1_accuracy']:.1%})")
    a(f"  Model top-1 accuracy vs pseudo-labels:     "
      f"{cv_model['top1_correct']:.0f}/{cv_model['orders']:.0f} ({cv_model['top1_accuracy']:.1%})")
    a("")
    a("FINAL MODEL FIT ON PSEUDO-LABELED CANDIDATES")
    a(f"  In-sample top-1 accuracy vs pseudo-labels: "
      f"{final_model['top1_correct']:.0f}/{final_model['orders']:.0f} ({final_model['top1_accuracy']:.1%})")
    a("")
    a("Caveat: these metrics measure agreement with automated pseudo-labels,")
    a("not independent inbound truth.")
    a("")
    a("Files written:")
    for path in [
        FULL_CANDIDATES_PATH,
        FEATURES_PATH,
        PSEUDO_LABELS_PATH,
        SCORED_CANDIDATES_PATH,
        MATCHES_PATH,
        MODEL_PATH,
    ]:
        a(f"  {os.path.basename(path)}")

    with open(EVAL_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"baseline_cv": baseline, "model_cv": cv_model, "final_model": final_model}


def main() -> None:
    features = build_features()
    features.to_csv(FEATURES_PATH, index=False)
    features.loc[features["pseudo_label"].eq(1)].to_csv(PSEUDO_LABELS_PATH, index=False)

    cv_scored = cross_validated_scores(features)
    final_model = train_logistic(features)
    final_scored = features.copy()
    final_scored["model_probability"] = predict_proba(final_model, final_scored)
    final_scored = final_scored.sort_values(
        ["dtc_row_id", "model_probability"], ascending=[True, False]
    )
    final_scored.to_csv(SCORED_CANDIDATES_PATH, index=False)

    best = _top_by(final_scored, "model_probability").copy()
    best = best.rename(columns={
        "ticket_row_id": "predicted_inbound_row_id",
        "ticket": "predicted_inbound_ticket",
        "model_probability": "inbound_match_probability",
    })
    keep = [
        "dtc_row_id", "dtc_order_no", "predicted_inbound_row_id",
        "predicted_inbound_ticket", "inbound_match_probability", "rule_score",
        "rule_rank", "pseudo_labeled_order", "pseudo_label", "pseudo_margin",
    ]
    best[keep].to_csv(MATCHES_PATH, index=False)

    metrics = write_evaluation(cv_scored, final_scored)
    save_model(final_model, metrics)

    print(f"Wrote {FULL_CANDIDATES_PATH}")
    print(f"Wrote {FEATURES_PATH}")
    print(f"Wrote {PSEUDO_LABELS_PATH}")
    print(f"Wrote {SCORED_CANDIDATES_PATH}")
    print(f"Wrote {MATCHES_PATH}")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {EVAL_PATH}")


if __name__ == "__main__":
    main()
