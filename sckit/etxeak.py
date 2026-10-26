import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("etxeak.csv")
X = df.drop(columns=["prezioa"])
y = df["prezioa"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
scaler_ss = StandardScaler()
X_train_ss = scaler_ss.fit_transform(X_train)
X_test_ss = scaler_ss.transform(X_test)
#print(X_train_ss.mean(axis=0))

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)
lr_ss = LinearRegression().fit(X_train_ss, y_train)

#print(f"LR Puntuazioa eskalatugabe: {round(lr.score(X_test, y_test), 4)}")
#print(f"LR Puntuazioa eskalatuta: {round(lr_ss.score(X_test_ss, y_test), 4)}")
#print("Koefizienteak (eskalatugabe):", lr.coef_)
#print("Koefizienteak (eskalatuta):", lr_ss.coef_)

knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
knn_ss = KNeighborsRegressor(n_neighbors=5).fit(X_train_ss, y_train)

#print(f"KNN Puntuazioa eskalatugabe: {round(knn.score(X_test, y_test), 4)}")
#print(f"KNN Puntuazioa eskalatuta: {round(knn_ss.score(X_test_ss, y_test), 4)}")

np.random.seed(42)

X_train["irrelevantea"] = np.random.randint(1, 5001)
X_test["irrelevantea"] = np.random.randint(1, 5001)

scaler_irr = StandardScaler()
X_train_irr_ss = scaler_irr.fit_transform(X_train)
X_test_irr_ss = scaler_irr.transform(X_test)

knn_irr_ss = KNeighborsRegressor(n_neighbors=5).fit(X_train_irr_ss, y_train)

#print(f"KNN Puntuazioa (zutabe irrelevantearekin): {round(knn_irr_ss.score(X_test_irr_ss, y_test), 4)}")

scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

lr_mm = LinearRegression().fit(X_train_mm, y_train)
knn_mm = KNeighborsRegressor(n_neighbors=5).fit(X_train_mm, y_train)

#print(f"LR Puntuazioa (MinMaxScaler): {round(lr_mm.score(X_test_mm, y_test), 4)}")
#print(f"KNN Puntuazioa (MinMaxScaler): {round(knn_mm.score(X_test_mm, y_test), 4)}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
eredua = LinearRegression().fit(X_train, y_train)
y_iragarpena = eredua.predict(X_test)
mae = mean_absolute_error(y_test, y_iragarpena)
mse = mean_squared_error(y_test, y_iragarpena)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_iragarpena)
print(mae, rmse, r2) 

try:
    azalera = float(input("Sartu ezazu azalera metro karratutan: "))
    gela = float(input("Sartu ezazu gela kopurua: "))
    urte = float(input("Sartu ezazu urte eraikia: "))

    etxe_berria = pd.DataFrame(
        [[azalera, gela, urte]],
        columns=["azalera_m2", "gela_kopurua", "urte_eraikia"],
    )
    pred_lr = lr.predict(etxe_berria)[0]
    etxe_berria_ss = scaler_ss.transform(etxe_berria)
    pred_lr_ss = lr_ss.predict(etxe_berria_ss)[0]
    pred_knn_ss = knn_ss.predict(etxe_berria_ss)[0]

    print(f"Erregresio linealaren prezioa: {round(pred_lr, 2)} €")
    print(f"KNN-arekin prezioa: {round(pred_knn_ss, 2)} €")

except ValueError:
    print("Etxeko balioak ez dira zuzenak")
