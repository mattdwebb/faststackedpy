"""Development-only comparison of native results; never a runtime dependency."""
import json
from pathlib import Path
import sys
import pandas as pd
from faststackedpy import fit

root=Path(sys.argv[1])
R=pd.read_csv(root/'faststackedR/tests/oracle_results.csv')
S=pd.read_csv(root/'faststackedstata/tests/oracle_results.csv')
errors=[]
for i,row in R.iterrows():
    d=pd.read_csv(root/f'faststackedR/tests/{row["mode"]}_{str(row.weighted).upper()}_{str(row.adjusted).upper()}.csv')
    p=fit(d,'y','time',unit='id',gvar='g',event_window=(-2,1),data_mode=row['mode'],
          covariates=['x','x0'] if row.adjusted else None,weights='w' if row.weighted else None)
    e={'mode':row['mode'],'weighted':bool(row.weighted),'adjusted':bool(row.adjusted)}
    for a,b in [('estimate','estimate'),('native_se','se'),('native_p','native_p')]:
        values=[p[a],row[b],S.iloc[i][b]]
        e[a+'_max_difference']=max(values)-min(values)
        assert e[a+'_max_difference']<1e-10
    errors.append(e)
print(json.dumps(errors,indent=2))
print('CROSS_LANGUAGE_PASS; maximum discrepancy =',max(max(v for k,v in e.items() if k.endswith('difference')) for e in errors))
