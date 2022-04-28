import lda_iris_ as ld
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

column = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# load the data

data = ld.data_to_pandas(url, column)

data.isnull().sum() # checking for null values

data.Class.unique() # unique ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


data['Target']=data['Class'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})


X,y=data.drop(['Class', 'Target'], axis=1).values, data['Target'].values 

X_train, X_test, y_train, y_test=ld.split_tt(X, y)

# transfroming the data 
scaler=StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.transform(X_test)


# train model

lda=ld.train_model_lda(X_train_scale, y_train) 

prediction=lda.predict(X_test_scale) # prediction

#Accuracy
print(f'Accuracy of the model :{accuracy_score(y_test, prediction)}')

#plotting the confusion matrix
confusion_ma =plot_confusion_matrix(lda, X_test_scale, y_test)
print(confusion_ma)