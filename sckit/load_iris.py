from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

df_iris = load_iris(as_frame=True)
X = df_iris.data
y = df_iris.target

print(X.shape)
print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42)
eredua = KNeighborsClassifier(n_neighbors=3)
eredua.fit(X_train, y_train)
print(eredua.score(X_test, y_test))