import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("etxeak.csv")

X = df[["azalera_m2"]]
y = df["prezioa"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

pf = PolynomialFeatures(degree=2)
X_tr_p = pf.fit_transform(X_tr)
X_te_p = pf.transform(X_te)

print(pf.get_feature_names_out())

eredua_pol = LinearRegression().fit(X_tr_p, y_tr)
print(eredua_pol.score(X_te_p, y_te)) # 0.988


for gradua in [1, 2, 3, 5, 8, 12]:
    pf = PolynomialFeatures(degree=gradua, include_bias=False)
    sc = StandardScaler()
    X_tr_pf = sc.fit_transform(pf.fit_transform(X_tr))
    X_te_pf = sc.transform(pf.transform(X_te))
    e = LinearRegression().fit(X_tr_pf, y_tr)
    print(gradua, round(e.score(X_tr_pf, y_tr), 3), # entrenamendua
        round(e.score(X_te_pf, y_te), 3)) # proba

from sklearn.model_selection import learning_curve
tamainak, tr, va = learning_curve(LinearRegression(), X, y, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv=5)
print(tamainak) # [ 6 12 19 25 32]
print(tr.mean(axis=1)) # [0.970 0.961 0.955 0.949 0.945]
print(va.mean(axis=1)) # [0.911 0.916 0.923 0.926 0.929]



sc =StandardScaler()
Xtr_sc = sc_fit_transform(X_train)
Xte_sc = sc.transform(X_test)

knnsc = KneighborsClassifier(n_neighbors= 5)
knnsc.fit(Xtr_sc, y_train)

print(f"StandardScaler: {knnsc.score(Xte_sc, y_test)}")