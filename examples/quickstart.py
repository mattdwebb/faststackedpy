from pathlib import Path
import pandas as pd
from faststackedpy import fit

data = pd.read_csv(Path(__file__).with_name("school_policy.csv"))
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
