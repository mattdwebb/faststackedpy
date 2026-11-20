"""Transparent saturated event-study WLS with original-cluster CR1."""
import numpy as np
import pandas as pd
from .probability import student_two_sided


class StackedError(ValueError):
    """Invalid input or an unidentified stacked design."""


def _require(ok, message):
    if not ok:
        raise StackedError(message)


def _point(d, lo, hi, mode, control, xs, native=True):
    events = np.arange(lo, hi + 1)
    cohorts = sorted(d.loc[np.isfinite(d.g), "g"].unique())
    stacks, dropped = [], []
    for a in cohorts:
        if a + lo < d.t.min() or a + hi > d.t.max():
            dropped.append(a)
            continue
        clean = (d.g > a + hi)
        if control == "never_treated":
            clean = np.isinf(d.g)
        elif control == "not_yet_treated":
            clean &= np.isfinite(d.g)
        s = d.loc[(d.g.eq(a) | clean) & d.t.between(a + lo, a + hi)].copy()
        s["a"], s["e"], s["z"] = a, s.t - a, (s.g == a).astype(float)
        if s.z.nunique() < 2:
            dropped.append(a)
            continue
        if not all(((s.e == e) & (s.z == z)).any() for e in events for z in [0, 1]):
            dropped.append(a)
            continue
        if mode == "panel":
            _require((s.groupby("u").size() == len(events)).all(),
                     "UNSUPPORTED: unbalanced panel within event window; use an explicit balanced sample")
            _require((s.groupby("u").w.nunique() == 1).all(),
                     "UNSUPPORTED: panel observation weights must be time invariant within a stack")
        stacks.append(s)
    _require(bool(stacks), "IDENTIFICATION_FAILURE: no identified cohorts in window")
    s = pd.concat(stacks, ignore_index=True)
    treated = s.w * s.z
    ta = treated.groupby(s.a).sum()
    shares = ta / ta.sum()
    s["q"] = s.w.astype(float)
    if mode == "panel":
        ca = (s.w * (1 - s.z)).groupby(s.a).sum()
        ratio = shares / (ca / ca.sum())
        s.loc[s.z == 0, "q"] *= s.loc[s.z == 0, "a"].map(ratio)
    else:
        counts = s.groupby(["a", "e", "z"]).w.sum()
        totals = s.groupby(["e", "z"]).w.sum()
        s["q"] = [w * shares[a] * totals[e, z] / counts[a, e, z]
                  for a, e, z, w in zip(s.a, s.e, s.z, s.w)]
    nonref = events[events != -1]
    E = (s.e.to_numpy()[:, None] == nonref).astype(float)
    X = np.column_stack([np.ones(len(s)), s.z, E, E * s.z.to_numpy()[:, None], s[xs]])
    root = np.sqrt(s.q.to_numpy())
    A = X * root[:, None]
    beta, _, rank, _ = np.linalg.lstsq(A, s.y.to_numpy() * root, rcond=None)
    n, k = X.shape
    _require(rank == k, "IDENTIFICATION_FAILURE: singular regression design")
    codes, clusters = pd.factorize(s.c, sort=False)
    G = len(clusters)
    V = None
    if native:
        _require(n > k, "IDENTIFICATION_FAILURE: no residual degrees of freedom")
        _require(G >= 2, "IDENTIFICATION_FAILURE: at least two original clusters required")
        resid = s.y.to_numpy() - X @ beta
        score = np.zeros((G, k))
        np.add.at(score, codes, X * (s.q.to_numpy() * resid)[:, None])
        bread = np.linalg.inv(A.T @ A)
        V = bread @ (score.T @ score) @ bread * G / (G - 1) * (n - 1) / (n - k)
        V = (V + V.T) / 2
    ix = 2 + len(nonref) + np.arange(len(nonref))
    L = np.zeros(k)
    L[ix[nonref >= 0]] = 1 / (hi + 1)
    estimate = float(L @ beta)
    se = float(np.sqrt(max(0, L @ V @ L))) if native else None
    p = student_two_sided(estimate / se, G - 1) if se is not None and se > 0 else None
    event = [{"event_time": int(e), "estimate": float(beta[j]),
              "native_se": float(np.sqrt(max(0, V[j, j]))) if native else None} for e, j in zip(nonref, ix)]
    event.append({"event_time": -1, "estimate": 0., "native_se": 0.})
    return dict(estimate=estimate, target_label=("regression-adjusted Q-weighted post-event contrast" if xs
                else "trimmed aggregate ATT: equal mean over post event times"),
                native_se=se, native_p=p, df=G - 1 if native else None, native_df=G - 1 if native else None,
                status="PASS", event_study=sorted(event, key=lambda r:r["event_time"]),
                event_covariance=V[np.ix_(ix, ix)] if native else None, event_covariance_times=nonref.tolist(),
                n_obs=len(d), n_used=s.row.nunique(), n_stacked=n, n_clusters=G,
                cohorts=list(shares.index), cohort_weights=shares.to_dict(), dropped_cohorts=dropped,
                metadata=dict(event_window=[lo, hi], baseline=-1, data_mode=mode,
                              control_group=control, covariates_used=xs,
                              covariate_timing="contemporaneous additive; no transformation",
                              native_vce="CR1: G/(G-1)*(Nstack-1)/(Nstack-K)",
                              native_reference="Student t, G-1: author Stata lincom / underlying fixest; R scalar helper CI uses normal",
                              oracle_commit="3116918a5773c6415acc9bce6699c5c15b3a1792"),
                stack=s[["row", "a", "e", "z", "q", "c"]].copy())


def fit(data, y, time, unit=None, gvar=None, treatment=None, cluster=None,
        covariates=None, event_window=None, data_mode="auto", **options):
    """Fit saturated WFH event study. See METHOD_SPECIFICATION.md for semantics.

    Options: weights, control_group, inference, assignments. Assignments are a
    DataFrame with columns equal to cluster IDs and rows containing cohort dates
    (NaN or +Inf for never). Only supplied, unique, count-preserving alternatives
    are accepted. Cluster cohort must be homogeneous for randomization.
    """
    allowed = {"weights", "control_group", "inference", "assignments"}
    unknown = set(options) - allowed
    _require(not unknown, "UNSUPPORTED options: " + ", ".join(sorted(unknown)))
    control = options.get("control_group", "both")
    _require(control in {"both", "never_treated", "not_yet_treated"}, "invalid control_group")
    mode = ("panel" if unit is not None else "rcs") if data_mode == "auto" else data_mode
    _require(mode in {"panel", "rcs"}, "data_mode must be panel, rcs, or auto")
    _require(mode != "panel" or unit is not None, "panel requires unit")
    cluster = cluster or unit
    _require(cluster is not None, "original cluster required, particularly for RCS")
    _require(event_window is not None and len(event_window) == 2, "event_window=(-pre,post) is required")
    lo, hi = event_window
    _require(np.isfinite([lo, hi]).all() and lo == int(lo) and hi == int(hi) and lo <= -1 and hi >= 0,
             "event_window must be integer, include -1 baseline and 0")
    lo, hi = int(lo), int(hi)
    covariates = [] if covariates is None else list(covariates)
    df = pd.DataFrame(data).reset_index(drop=True)
    _require(len(df) > 0, "empty data")
    d = pd.DataFrame({"y": df[y], "t": df[time], "c": df[cluster], "row": np.arange(len(df))})
    d["u"] = df[unit] if unit else d.row
    d["w"] = df[options["weights"]] if options.get("weights") else 1.
    xs = ["x" + str(i) for i in range(len(covariates))]
    for name, x in zip(xs, covariates):
        d[name] = df[x]
    _require(not d.isna().any().any(), "missing values in outcome/time/IDs/weights/covariates")
    _require(np.isfinite(d[["y", "t", "w"] + xs].to_numpy(dtype=float)).all(), "nonfinite numeric input")
    _require((d.w > 0).all() and (d.t == np.floor(d.t)).all(), "positive weights and integer time required")
    if mode == "panel":
        _require(not d.duplicated(["u", "t"]).any(), "duplicate unit-time rows")
        _require((d.groupby("u").c.nunique() == 1).all(), "panel cluster must be constant within unit")
    _require(gvar is not None or treatment is not None, "gvar or treatment required")
    if gvar:
        d["g"] = pd.to_numeric(df[gvar], errors="raise").fillna(np.inf)
        _require((np.isposinf(d.g) | (np.isfinite(d.g) & (d.g == np.floor(d.g)))).all(), "invalid cohort dates")
    else:
        _require(mode == "panel", "UNSUPPORTED: RCS treatment-only input; supply cohort dates")
        D = df[treatment]
        _require(D.isin([0, 1]).all(), "treatment must be binary")
        first = d.loc[D == 1].groupby("u").t.min()
        d["g"] = d.u.map(first).fillna(np.inf)
        _require(not (d.g == d.groupby("u").t.transform("min")).any(),
                 "cannot infer adoption for initially treated units; supply gvar")
    if mode == "panel":
        _require((d.groupby("u").g.nunique() == 1).all(), "cohort must be constant within panel unit")
    if treatment:
        _require(df[treatment].isin([0, 1]).all() and np.array_equal(df[treatment], (d.t >= d.g).astype(int)),
                 "inconsistent or nonabsorbing treatment history")
    methods = options.get("inference", "native")
    methods = [methods] if isinstance(methods, str) else list(methods)
    _require(set(methods) <= {"point", "native", "jackknife", "ri_beta", "ri_t_native"}, "UNSUPPORTED inference")
    _require("point" not in methods or methods == ["point"], "point cannot be combined with inference methods")
    ri = set(methods) & {"ri_beta", "ri_t_native"}
    result = _point(d, lo, hi, mode, control, xs, native=methods != ["point"])
    result["events"] = result["event_study"]
    result["metadata"]["covariates_requested"] = covariates
    result["metadata"]["covariates_used"] = covariates

    def jack(data, obs):
        values = []
        ids = data.c.unique()
        for c in ids:
            r = _point(data.loc[data.c != c], lo, hi, mode, control, xs, native=False)
            values.append(r["estimate"])
        values = np.asarray(values)
        se = float(np.sqrt((len(ids) - 1) / len(ids) * np.sum((values - values.mean()) ** 2)))
        _require(se > 1e-14, "IDENTIFICATION_FAILURE: zero jackknife variance")
        return dict(se=se, p=student_two_sided(obs["estimate"] / se, len(ids) - 1),
                    df=len(ids)-1, estimates=values.tolist(), clusters=ids.tolist())

    if "jackknife" in methods:
        result["jackknife"] = jack(d, result)
        result["jk_se"], result["jk_p"] = result["jackknife"]["se"], result["jackknife"]["p"]
    if ri:
        assignments = options.get("assignments")
        _require(isinstance(assignments, pd.DataFrame), "UNSUPPORTED: RI requires supplied assignment DataFrame")
        _require((d.groupby("c").g.nunique() == 1).all(), "RI requires homogeneous cohort within assignment cluster")
        ids = d.c.unique()
        _require(len(assignments.columns) == len(ids) and set(assignments.columns) == set(ids), "assignment columns must exactly match original clusters")
        observed = d.groupby("c").g.first().reindex(ids).to_numpy()
        arr = assignments.loc[:, ids].to_numpy(dtype=float)
        arr = np.where(np.isnan(arr), np.inf, arr)
        _require(len(arr) > 0 and len(np.unique(arr, axis=0)) == len(arr), "assignments must be nonempty and unique")
        _require(all(np.array_equal(np.sort(a), np.sort(observed)) and not np.array_equal(a, observed) for a in arr),
                 "assignments must preserve cohort counts and exclude observed assignment")
        stats = {m: [] for m in ri}
        for a in arr:
            dd = d.copy()
            dd["g"] = dd.c.map(dict(zip(ids, a)))
            rr = _point(dd, lo, hi, mode, control, xs)
            for m in ri:
                denom = 1.
                if m == "ri_t_native":
                    denom = rr["native_se"]
                _require(denom > 1e-14, "IDENTIFICATION_FAILURE: zero randomized SE")
                stats[m].append(rr["estimate"] / denom)
        for m in ri:
            den = result["native_se"] if m == "ri_t_native" else 1.
            _require(den > 1e-14, "IDENTIFICATION_FAILURE: zero observed SE")
            obs = result["estimate"] / den
            # Inclusive comparisons allow only floating-point roundoff near ties.
            p = (1 + sum(abs(v) >= abs(obs) - 64*np.finfo(float).eps*max(1.,abs(v),abs(obs))
                         for v in stats[m])) / (len(arr) + 1)
            result[m] = dict(p=p, statistic=obs, randomized_statistics=stats[m], reps=len(arr), null=0.)
            result[m + "_p"] = p
    elif options.get("assignments") is not None:
        raise StackedError("assignments supplied without an RI method")
    return result
