import itertools
import numpy as np
import pandas as pd
import pytest
from faststackedpy import fit, StackedError


def fixture(rcs=False):
    rows = []
    for u in range(12):
        g = 3 if u < 4 else (5 if u < 8 else np.inf)
        for t in range(1, 8):
            for j in range(1 + ((u + t) % 3 if rcs else 0)):
                x = np.sin(u * 2.3 + t * .8 + j)
                rows.append((u, t, g, u*.3 + t*.2 + (t>=g)*(1+u*.2) + .3*x + .2*np.cos(u*t+j), x, u%2, 1+u/20))
    return pd.DataFrame(rows, columns=['id','time','g','y','x','x0','w'])


def run(d, **kw):
    args = dict(y='y', time='time', unit='id', gvar='g', event_window=(-2,1))
    args.update(kw)
    return fit(d, **args)


def test_hand_weights():
    d = pd.DataFrame([(u,t,2012 if u==1 else (2014 if u==2 else np.inf),u+t*.1)
                      for u in range(1,5) for t in range(2010,2016)], columns=['id','time','g','y'])
    s = run(d,event_window=(-1,1))['stack']
    assert len(s)==21
    assert np.allclose(s.loc[(s.a==2012)&(s.z==0),'q'],5/6)
    assert np.allclose(s.loc[(s.a==2014)&(s.z==0),'q'],5/4)
    assert np.all(s.loc[s.z==1,'q']==1)


@pytest.mark.parametrize('rcs',[False,True])
@pytest.mark.parametrize('weighted',[False,True])
def test_explicit_wls(rcs, weighted):
    d=fixture(rcs)
    r=run(d,data_mode='rcs' if rcs else 'panel',covariates=['x','x0'],weights='w' if weighted else None)
    s=r['stack']; z=s.z.to_numpy(); ev=s.e.to_numpy()
    X=np.column_stack([np.ones(len(s)),z]+[(ev==e)*1. for e in [-2,0,1]]+
                      [z*(ev==e) for e in [-2,0,1]]+[d.iloc[s.row][c].to_numpy() for c in ['x','x0']])
    q=s.q.to_numpy(); Y=d.iloc[s.row].y.to_numpy()
    b=np.linalg.solve(X.T@(q[:,None]*X),X.T@(q*Y))
    assert r['estimate']==pytest.approx((b[6]+b[7])/2,abs=1e-10)
    assert r['native_se']>0
    assert r['n_clusters']==12


def test_linearity_and_no_mutation():
    d=fixture(); original=d.copy(deep=True)
    a=run(d,covariates=['x'])['estimate']
    d2=d.assign(y=np.cos(d.y)); b=run(d2,covariates=['x'])['estimate']
    d3=d.assign(y=2*d.y-.7*d2.y)
    assert run(d3,covariates=['x'])['estimate']==pytest.approx(2*a-.7*b,abs=1e-10)
    pd.testing.assert_frame_equal(d,original)


def test_jackknife_rebuild():
    d=fixture(); r=run(d,inference='jackknife')
    vals=[run(d[d.id!=c])['estimate'] for c in d.id.unique()]
    assert r['jackknife']['estimates']==pytest.approx(vals)
    assert r['jk_se']==pytest.approx(np.sqrt(11/12*np.sum((vals-np.mean(vals))**2)))


def assignments(d):
    g=d.groupby('id').g.first().to_numpy()
    return pd.DataFrame([np.roll(g,k) for k in [1,2,3]],columns=range(12))


def test_ri_rebuild():
    d=fixture(); aa=assignments(d)
    r=run(d,inference=['ri_beta','ri_t_native'],assignments=aa)
    stats=[]
    for _,row in aa.iterrows():
        rr=run(d.assign(g=d.id.map(row)),inference='native')
        stats.append(rr['estimate']/rr['native_se'])
    assert r['ri_t_native']['randomized_statistics']==pytest.approx(stats)
    with pytest.raises(StackedError,match='unique'):
        run(d,inference='ri_beta',assignments=pd.concat([aa,aa]))


@pytest.mark.parametrize('option', ['fe','reps','seed','weight_type','null'])
def test_unsupported(option):
    with pytest.raises(StackedError,match='UNSUPPORTED'):
        run(fixture(),**{option:1})


def test_bad_panel_and_history():
    d=fixture()
    with pytest.raises(StackedError,match='unbalanced'):
        run(d.drop(index=1))
    with pytest.raises(StackedError,match='singular'):
        run(d.assign(const=1),covariates=['const'])
    with pytest.raises(StackedError,match='inconsistent'):
        run(d.assign(D=0),treatment='D')
    with pytest.raises(StackedError,match='missing'):
        run(d.assign(y=np.nan))


def test_treatment_inference():
    d=fixture().assign(D=lambda x:(x.time>=x.g).astype(int))
    assert run(d,gvar=None,treatment='D')['estimate']==pytest.approx(run(d)['estimate'])


def test_singleton_deletion_rebuilds_remaining_stacks():
    d=fixture(); d=d[(d.id==0)|(d.id>=4)]
    jk = run(d,inference='jackknife')['jackknife']
    expected = [run(d[d.id!=i],inference='point')['estimate'] for i in jk['clusters']]
    np.testing.assert_allclose(jk['estimates'],expected)


def test_homogeneous_effect():
    d=fixture(); d['y']=.4*d.id+.3*d.time+2*(d.time>=d.g)
    assert run(d)['estimate']==pytest.approx(2,abs=1e-10)


def test_point_with_one_cluster():
    d=fixture().assign(cluster=1)
    r=run(d,cluster='cluster',inference='point')
    assert r['estimate']==pytest.approx(run(d)['estimate'])
    assert r['native_se'] is None and r['native_df'] is None
    assert r['events']==r['event_study']
    with pytest.raises(StackedError,match='two original clusters'):
        run(d,cluster='cluster')
