#Aztertu R² eta beste metrika bat erabiliz, zein puntuazio lortzen duten KNN ezberdinak, 
# 1 eta 6 bizilagun artean, iragartzen duena zenbat logela dauzkan etxe bat, prezioa eta azalera emanda. 
# Hausnartu non dagoen overfit, non orekatua eta non underfit.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("etxeak.csv")
X = df[["azalera_m2", "prezioa"]]
y = df["gela_kopurua"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

for i in range(6):
    knn = KNeighborsRegressor(n_neighbors=i+1)
    knn.fit(X_tr_sc, y_tr)

    y_iragarpena = knn.predict(X_te_sc)
    mae = mean_absolute_error(y_te, y_iragarpena)
    r2 = r2_score(y_te, y_iragarpena)

    y_tr_iragarpena = knn.predict(X_tr_sc)
    mae_tr = mean_absolute_error(y_tr, y_tr_iragarpena)
    r2_tr = r2_score(y_tr, y_tr_iragarpena)
    print(f"{i+1}. {mae_tr, mae, r2_tr, r2}")  
