from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
import pytest
from faststackedpy import fit

ROOT=Path(__file__).parent/'fixtures'


@pytest.mark.parametrize('i',range(8))
def test_frozen_official_results(i):
    r=pd.read_csv(ROOT/'oracle_results.csv').iloc[i]
    name=f'{r["mode"]}_{str(r.weighted).upper()}_{str(r.adjusted).upper()}.csv'
    d=pd.read_csv(ROOT/name)
    o=fit(d,'y','time',unit='id',gvar='g',event_window=(-2,1),data_mode=r['mode'],
          weights='w' if r.weighted else None,covariates=['x','x0'] if r.adjusted else None)
    assert o['estimate']==pytest.approx(r.estimate,abs=1e-10)
    assert o['native_se']==pytest.approx(r.se,abs=1e-10)
    assert o['native_p']==pytest.approx(r.native_p,abs=1e-12)
    assert o['native_df']==r.native_df


def test_r_inference():
    d=pd.read_csv(ROOT/'panel_FALSE_FALSE.csv')
    g=d.groupby('id').g.first().to_numpy()
    aa=pd.DataFrame([np.roll(g,k) for k in [1,2,3]],columns=range(12))
    r=fit(d,'y','time',unit='id',gvar='g',event_window=(-2,1),
          inference=['jackknife','ri_beta','ri_t_native'],assignments=aa)
    expected=pd.read_csv(ROOT/'inference_results.csv').set_index('method').value
    assert r['native_se']==pytest.approx(expected['native'],abs=1e-10)
    assert r['jk_se']==pytest.approx(expected['jackknife'],abs=1e-10)
    for m in ['ri_beta','ri_t_native']:
        assert r[m+'_p']==expected[m]


def worker(_):
    d=pd.read_csv(ROOT/'panel_FALSE_FALSE.csv')
    return fit(d,'y','time',unit='id',gvar='g',event_window=(-2,1))['estimate']


def test_spawn_process_isolation():
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        values=list(pool.map(worker,range(4)))
    assert values==pytest.approx([worker(0)]*4)
