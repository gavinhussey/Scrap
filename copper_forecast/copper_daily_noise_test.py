"""
copper_daily_noise_test.py — is the top-10%-conviction Dstat (~0.58) real or noise?

Two repeated experiments on the same daily cross-asset pipeline:

  REAL  : rerun the walk-forward + top-decile conviction filter many times under
          different fold layouts (init-train fractions) and ensemble seeds. Shows
          how much the 0.58 bounces around on genuine data.

  NULL  : same pipeline, but the up/down labels are SHUFFLED so there is zero
          predictable signal by construction. Repeated many times, this gives the
          distribution of top-decile Dstat you'd get from pure luck.

If REAL overlaps NULL, the 0.58 was noise. The empirical p-value = fraction of
NULL runs that match or beat the REAL mean.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from copper_daily_direction import download, build_features, CALIB_FRAC

TOP_COV = 0.10
SEEDS = 5
FOLDS = 5


def ensemble_predict(Xtr, ytr, seed_offset):
    models = [
        HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, early_stopping=True, random_state=s + seed_offset)
        .fit(Xtr, ytr)
        for s in range(SEEDS)
    ]
    return lambda Z: np.mean([m.predict_proba(Z)[:, 1] for m in models], axis=0)


def run_once(Xv, yv, init_frac, seed_offset, top_cov=TOP_COV):
    """Pooled top-conviction Dstat for one walk-forward pass."""
    T = len(yv)
    first = int(T * init_frac)
    block = (T - first) // FOLDS
    selected = []
    for k in range(FOLDS):
        lo = first + k * block
        hi = first + (k + 1) * block if k < FOLDS - 1 else T
        if hi <= lo:
            continue
        cut = int(lo * (1 - CALIB_FRAC))
        predict = ensemble_predict(Xv[:cut], yv[:cut], seed_offset)
        conv_calib = np.abs(predict(Xv[cut:lo]) - 0.5)
        p_te = predict(Xv[lo:hi])
        conv_te = np.abs(p_te - 0.5)
        correct = ((p_te >= 0.5).astype(int) == yv[lo:hi])
        thr = np.quantile(conv_calib, 1 - top_cov)
        selected.append(correct[conv_te >= thr])
    c = np.concatenate(selected)
    return c.mean(), len(c)


def summarize(name, vals):
    v = np.array(vals)
    print(f"{name:5s}  n_runs={len(v):3d}  mean={v.mean():.4f}  std={v.std():.4f}  "
          f"min={v.min():.4f}  p95={np.quantile(v,0.95):.4f}  max={v.max():.4f}")
    return v


def main():
    print("Downloading daily data (copper + cross-asset)...")
    X, y, _ = build_features(download())
    Xv, yv = X.to_numpy(float), y.to_numpy(int)
    print(f"Samples: {len(yv)} days | up-share: {yv.mean():.3f} | top coverage: {TOP_COV}\n")

    # REAL: vary fold layout and ensemble seed -> genuine spread on true labels.
    real = []
    for init_frac in (0.45, 0.475, 0.50, 0.525, 0.55):
        for off in (0, 100, 200, 400):
            d, n = run_once(Xv, yv, init_frac, off)
            real.append(d)
    print("--- distributions of top-10% conviction Dstat ---")
    rv = summarize("REAL", real)

    # NULL: shuffled labels -> no signal possible, repeated.
    rng = np.random.default_rng(0)
    null = []
    for i in range(80):
        y_perm = rng.permutation(yv)
        d, n = run_once(Xv, y_perm, 0.50, 0)
        null.append(d)
    nv = summarize("NULL ", null)

    # Empirical significance.
    p_mean = np.mean(nv >= rv.mean())
    p_058 = np.mean(nv >= 0.58)
    print(f"\nP(noise >= REAL mean {rv.mean():.4f}) = {p_mean:.3f}")
    print(f"P(noise >= 0.58)             = {p_058:.3f}")
    print(f"NULL is centered near {nv.mean():.3f} (the no-signal baseline).")
    print("\nVerdict: if REAL mean sits inside the NULL spread and p is large, "
          "the 0.58 was luck.")


if __name__ == "__main__":
    main()
