from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

df = load_iris()

# create feature and target arrays
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# define the model

model = KNeighborsClassifier(n_neighbors=7)

model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.score(X_test, y_test))
