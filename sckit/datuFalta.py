import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

etxeak = pd.read_csv("etxeak.csv")
X = etxeak[["azalera_m2", "gela_kopurua", "urte_eraikia"]]
y = etxeak["prezioa"]


Xn = X.astype(float)
r2 = np.random.default_rng(7)
Xn.loc[r2.choice(len(Xn), 8, replace=False), "azalera_m2"] = np.nan
Xn.loc[r2.choice(len(Xn), 6, replace=False), "urte_eraikia"] = np.nan
print(Xn.isna().sum().to_dict()) # {'azalera_m2': 8, 'gela_kopurua': 0, 'urte_eraikia': 6}
print(Xn)
a, b, c, d = train_test_split(Xn, y, test_size=0.2, random_state=42)

inputatzailea = SimpleImputer(strategy="most_frequent")
a_inp = inputatzailea.fit_transform(a)
b_inp = inputatzailea.transform(b)
print(np.isnan(a_inp).sum())

try:
    #lr = LinearRegression().fit(a, c)
    lr_inp = LinearRegression().fit(a_inp, c).score(b_inp, d)
    print(round(lr_inp, 4))
except ValueError as e:
    print(str(e).split(".")[0])