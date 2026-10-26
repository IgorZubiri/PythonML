import numpy as np
import pandas as pd

r = np.random.default_rng(3)
m = 200
adina = r.integers(18, 26, m)
ikasgela = r.choice(["A", "B", "C"], m)
efektua = pd.Series(ikasgela).map({"A": 0.0, "B": 2.0, "C": -1.5}).to_numpy()
nota = (5 + 0.15*(adina - 20) + efektua + r.normal(0, 0.5, m)).round(1)
ik = pd.DataFrame({"adina": adina, "ikasgela": ikasgela, "nota": nota})
ik.to_csv("ikasleak.csv", index=False)
print(ik.groupby("ikasgela").nota.mean().round(2).to_dict()) # {'A': 5.28, 'B':6.96, 'C': 3.87}