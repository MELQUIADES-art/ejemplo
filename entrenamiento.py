#Capitulo 10
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import pandas as pd
iris = load_iris(as_frame=True)
dt=pd.DataFrame(iris)
dt.head(5)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0) # Iris setosa
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)
y_pred
dt[dt["petal length (cm)"]==2]]
