import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('heart disease.csv')
x = data.drop("target", axis=1)
y = data["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)
res= classifier.predict(x_test)
print(res)
