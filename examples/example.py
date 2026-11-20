import pandas as pd
from faststackedpy import fit

data = pd.DataFrame([(u, t, 3 if u < 4 else (5 if u < 8 else None))
                     for u in range(12) for t in range(1, 8)], columns=["id", "time", "g"])
data["y"] = .3*data.id + .2*data.time + 2*(data.time >= data.g)
r = fit(data, "y", "time", unit="id", gvar="g", event_window=(-2, 1))
print({key: r[key] for key in ["estimate", "native_se", "target_label"]})
