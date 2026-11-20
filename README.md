# faststackedpy

**Corrective-weighted stacked DiD in Python.** Compare policy cohorts in separate clean event windows, then combine them using the corrective weights of Wing, Freedman and Hollingsworth.

Stacked DiD builds a separate sample around each adoption cohort. It keeps comparison observations that remain untreated through the chosen window and uses corrective weights before estimating and aggregating the event-time effects. The window is part of the estimand, not just a plotting option.

## Install and run

Use Python 3.10 or newer. Install directly from GitHub; this package is not distributed on PyPI. Clone the repository
(or download and extract its ZIP), then run these commands from its root:

```sh
git clone https://github.com/mattdwebb/faststackedpy.git
cd faststackedpy
python -m pip install "faststackedpy[examples] @ git+https://github.com/mattdwebb/faststackedpy.git"
python examples/quickstart.py
```

The installer supplies the numerical dependencies and pandas for this example.
No R or Stata installation is required. A virtual environment is recommended
when installing this package. Record the source commit for reproducible work.

## A school-policy example

The example has **960 pupils in 12 schools over eight years**. Different pupils are observed each year. The outcome is a test score and the covariate is a predetermined baseline score. Four schools adopt a policy in 2013, four in 2015, and four never adopt. The outcome is in score points, and the inference cluster is the school. These are fictional teaching data, not evidence about an actual school policy. The [data guide](examples/DATA_GUIDE.md) defines every column and links the deterministic generator.

The complete calculation in `examples/quickstart.py` is:

```python
import pandas as pd
from faststackedpy import fit

data = pd.read_csv("examples/school_policy.csv")
data["first_treat"] = data["first_treat"].replace(0, float("nan"))
result = fit(data, "test_score", "year", gvar="first_treat", cluster="school",
             covariates=["baseline_score"], data_mode="rcs",
             event_window=(-1, 1), inference=["native", "jackknife"])
estimate, native_se = result["estimate"], result["native_se"]
jk_se, jk_p = result["jk_se"], result["jk_p"]
print(f"Estimate: {estimate:.6f} score points")
print("Native clustered SE: " + (f"{native_se:.6f}" if native_se is not None else "not available"))
print(f"Delete-school jackknife SE: {jk_se:.6f}")
print(f"Jackknife p-value: {jk_p:.6f}")
print("Schools: 12; jackknife reference: t(11)")
```

Actual output (rounded to six decimal places):

```text
Estimate: 1.771243 score points
Native clustered SE: 0.251554
Delete-school jackknife SE: 0.285132
Jackknife p-value: 0.000066
Schools: 12; jackknife reference: t(11)
```

The adjusted estimate is **1.77 score points**. The delete-school standard error is **0.29 points**: it measures how sensitive the estimate is to leaving out schools one at a time. The reported jackknife p-value tests a zero effect using Student t with 11 degrees of freedom. The native and jackknife standard errors are different calculations around the same point estimate; a larger or smaller standard error is not automatically the better one.

The generator adds a two-point policy effect, along with school effects, year
effects and varying disturbances. The fitted estimate need not be exactly two.
This example teaches the interface; it does not measure test size or establish
the causal assumptions needed for an application.

## Adapt the analysis

Change `covariates=["baseline_score"]` to `covariates=[]` for an unadjusted specification. For your own data, replace the CSV and column names, keep a constant adoption year within each school, and check the supported input rules below. Numeric covariates must be prepared before fitting.

Save the compact result, for example:

```python
pd.DataFrame([{"estimate": estimate, "native_se": native_se,
               "jackknife_se": jk_se, "jackknife_p": jk_p}]).to_csv(
    "policy_results.csv", index=False)
```

Use `inference="native"` for the analytic clustered result or
`inference="jackknife"` for deletion inference alone. `result["event_study"]`
contains the supported event-time estimates. Changing `event_window` changes
which effects enter the reported aggregate.

Randomization methods accept an explicit, validated assignment matrix. See the
[inference guide](INFERENCE.md); the standalone package does not silently invent
an assignment rule. The [Python DiD Suite](https://github.com/mattdwebb/didsuitepy)
can generate shared assignments for a supported comparison.


## Supported scope and failures

Both panels and repeated cross sections are supported. An explicit window containing event time −1 is required. In the standalone API, missing or positive-infinite cohort values mean never treated; zero is a genuine date. The example converts the CSV’s zero convention explicitly. Repeated cross sections require the original inference-cluster column and must not be relabelled as individual panels.

The jackknife uses every input school deletion. If a cohort disappears, the
estimator recomputes the supported aggregate and its weights over surviving
estimable effects. It does not discard failed deletions to produce a standard
error. A remaining aggregate must still be identified. Estimates from different
DiD methods need not target the same weighted effect when treatment effects vary.

## Help, references and checks

The [release-candidate validation record](RELEASE_VALIDATION.md) gives the exact

See [method](METHOD_SPECIFICATION.md), [inference](INFERENCE.md), [validation](VALIDATION.md), and [paper/source attribution](THIRD_PARTY_NOTICES.md). These pages contain the detailed estimator and inference
conventions. Report reproducible problems through the repository’s
[issue tracker](https://github.com/mattdwebb/faststackedpy/issues), including package
version, a small synthetic example and the full error message.

To run the numerical tests and the executable README check:

```sh
git clone https://github.com/mattdwebb/faststackedpy.git
cd faststackedpy
python -m pip install "faststackedpy[test,examples] @ git+https://github.com/mattdwebb/faststackedpy.git"
python -m pytest -q
```

The [MIT license](LICENSE) applies to this implementation; linked provenance
notes identify external methods and validation software. Cite the underlying
estimator as well as the software revision used.
