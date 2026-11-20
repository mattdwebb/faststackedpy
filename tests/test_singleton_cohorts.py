from pathlib import Path
import numpy as np
import pandas as pd
from faststackedpy import fit

def test_singleton_cohorts_rebuild_weights_and_surviving_support():
    root=Path(__file__).parent
    data=pd.read_csv(root/"singleton_cohorts.csv")
    expected=pd.read_csv(root/"singleton_reference.csv")
    for adjusted in (0,1):
        args=dict(y="y",time="time",gvar="g",cluster="state",data_mode="rcs",event_window=(-2,1),covariates=["x"] if adjusted else None,weights="w")
        result=fit(data,**args,inference="jackknife")
        physical=[fit(data[data.state!=i],**args,inference="point")["estimate"] for i in sorted(data.state.unique())]
        reference=expected[expected.adjusted==adjusted]
        np.testing.assert_allclose(result["jackknife"]["estimates"],physical,atol=1e-11,rtol=0)
        np.testing.assert_allclose(physical,reference.estimate.iloc[1:],atol=1e-10,rtol=0)
        assert abs(result["estimate"]-reference.estimate.iloc[0])<1e-10
        assert abs(result["jk_se"]-reference.jk_se.iloc[0])<1e-10
        assert result["jackknife"]["df"]==7
