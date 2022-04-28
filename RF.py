import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# load dataset
iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

# split the data into x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(data.values, target, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# initialize the model

clf = RandomForestClassifier()

clf.fit(x_train_scaled, y_train)

# predict

y_preds = clf.predict(x_test_scaled)

# accuracy score

acc = accuracy_score(y_test, y_preds)
print(acc) # 1.0

# plot confusion matrix

cm = confusion_matrix(y_test, y_preds)
cmd  = ConfusionMatrixDisplay(cm, display_labels=['setosa', 'versicolor', 'virginica'])
cmd.plot()
plt.show()