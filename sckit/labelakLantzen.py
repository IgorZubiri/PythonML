import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ik = pd.read_csv("ikasleak.csv")
le = LabelEncoder()
ik["ikasgela_kod"] = le.fit_transform(ik["ikasgela"])
#print(ik)
#print(le.classes_) # ['A' 'B' 'C'] (A -> 0, B -> 1, C -> 2)
Xl, yl = ik[["adina", "ikasgela_kod"]], ik["nota"]
a, b, c, d = train_test_split(Xl, yl, test_size=0.2, random_state=42)
lr = LinearRegression().fit(a, c)
print(round(lr.score(b, d), 3)) # 0.247

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
M = ohe.fit_transform(ik[["ikasgela"]]).toarray()
print(ohe.get_feature_names_out()) # ['ikasgela_A' 'ikasgela_B' 'ikasgela_C']
print(M[:3])

Xo = pd.concat([ik[["adina"]], pd.get_dummies(ik["ikasgela"]).astype(int)], axis=1)
a, b, c, d = train_test_split(Xo, ik["nota"], test_size=0.2, random_state=42)
print(Xo)
mo = LinearRegression().fit(a, c)
print(round(mo.score(b, d), 3)) # 0.787
print(pd.Series(mo.coef_, index=Xo.columns).round(2).to_dict())
print(mo.intercept_)