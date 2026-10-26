import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("switch2_precios.csv")
x = df.iloc[1:, 0] #eguna
y = df.iloc[1:, 1] #prezioa

X_media = x.mean()
Y_media = y.mean()
n_elementos = len(x)

numerador_1 = (x * y).sum()
numerador_1_b = 0

for i in range(n_elementos):
    numerador_1_b += x.iloc[i] * y.iloc[i]

print(numerador_1)
print(numerador_1_b)

numerador_2 = n_elementos * X_media * Y_media

numerador = numerador_1_b - numerador_2

denominador_1 = (x**2).sum()
denominador_1_b = 0
for i in range(n_elementos):
    denominador_1_b += x.iloc[i] ** 2

denominador_2 = n_elementos * (X_media ** 2)
denominador = denominador_1_b - denominador_2
A = numerador / denominador
print(A)

B = Y_media - A * X_media
print(B)

eguna = int(input("emaidazu egun bat: "))
prezio_estimatua = lambda x: B + x * A
print(f" {eguna} egunean prezio estimatua {prezio_estimatua} da")

plt.scatter(x, y)
plt.plot(x, prezio_estimatua(x), "r")
plt.show()

#para valorar el modelo
# Predicciones sobre el conjunto de entrenamiento
y_pred = A * x + B

# Metricas
mse = ((y - y_pred ) ** 2).mean()
rmse = np.sqrt(mse)
mae = (y - y_pred).abs().mean()
ss_res = (( y - y_pred ) ** 2) . sum ()
ss_tot = (( y - y . mean () ) ** 2) . sum ()
r2 = 1 - ss_res / ss_tot

print(f"MSE = {mse :.4f} ")
print(f"RMSE = {rmse :.4f} ")
print(f"MAE = {mae :.4f} ")
print(f"R2 = {r2 :.4f} ")

X_train_sk = x.values.reshape(-1, 1)
from sklearn.linear_model import LinearRegression

modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train_sk, y)

print(modelo_regresion.coef_[0])
print(A)
print(modelo_regresion.intercept_)
print(B)