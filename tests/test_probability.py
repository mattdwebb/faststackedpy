from pathlib import Path
import pandas as pd
import pytest
from faststackedpy.probability import student_two_sided

REF=pd.read_csv(Path(__file__).parent/'fixtures'/'probability_oracle.csv')


@pytest.mark.parametrize('i', range(42))
def test_r_pt(i):
    row=REF.iloc[i]
    assert student_two_sided(row.statistic,row.df)==pytest.approx(row.p,rel=1e-11,abs=1e-14)
    assert student_two_sided(-row.statistic,row.df)==pytest.approx(row.p,rel=1e-11,abs=1e-14)
