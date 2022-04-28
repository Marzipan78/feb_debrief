import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

"""
Using Knn to to make prediction for classes of Iris dataset 

"""

# Getting the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

#Encoding the target variable
oe = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1,dtype=np.int64)
irisdata.Class = oe.fit_transform(irisdata.loc[:, ['Class']])

#getting values of x and y 
X,y= irisdata.drop(['Class'],axis=1), irisdata['Class']

#splitting the dataset 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2,random_state=0,stratify=y)

#Data preprocessing 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Ploting error curves of differnt values of K for the train and test data; this will be used to choose the best k
error1= []
error2= []
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred1= knn.predict(X_train)
    error1.append(np.mean(y_train!= y_pred1))
    y_pred2= knn.predict(X_test)
    error2.append(np.mean(y_test!= y_pred2))

plt.plot(range(1,15),error1,label="train")
plt.plot(range(1,15),error2,label="test")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show() ## we need to chhose the k where the test error is low 

# or 

# #we can also choose k using the plot of accurarcies  against various k values. Here its 7 
acc_score= []
error2= []
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred= knn.predict(X_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    acc_score.append(acc)

plt.plot(range(1,15),acc_score)
plt.xlabel('k Value')
plt.ylabel('Accurarcy score')
plt.show() ## we need to choose the of k where there seems to be no changes in the graph. Here its 7


#Now we will use the choosen k value to fit our model and make prediction
knn_7= KNeighborsClassifier(n_neighbors=7)
knn_7.fit(X_train,y_train)
y_pred= knn_7.predict(X_test)
print('The accuracury is:', metrics.accuracy_score(y_test,y_pred))
plot_confusion_matrix(knn_7, X_test,y_test)
plt.show()




