# Importing packages
from operator import irshift, mod
from statistics import mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
#loading the dataset
iris = load_iris()

def dataframe(iris):
    #creating dataframe
    idf = pd.DataFrame(iris.data, columns=iris.feature_names)
    #Adding Target and class into dataframe
    idf['target'] = iris.target
    idf['class'] = iris.target_names[iris.target]
    # Removing 'cm' from columns
    idf.columns = [col.replace('(cm)', '').strip() for col in idf.columns]

    return idf

df = dataframe(iris)



#defining feauture and target
def feat_tar(df):
    X, y = df.drop(['class', 'target'], axis=1).values, df['target'].values
    return X, y

ft = feat_tar(df)



#Splitting the data
def split(ft):
    X_train, X_test, y_train, y_test = train_test_split(ft[0], ft[1], test_size = 0.3, random_state=0, stratify=ft[1])
    
    return X_train, y_train, X_test, y_test
sp = split(ft)



#Scale the data
def scale(sp):
    scaler = StandardScaler()
    X_train_scaled  = scaler.fit_transform(sp[0])
    X_test_scaled = scaler.transform(sp[2])
    return X_train_scaled, X_test_scaled
sc = scale(sp)



# Applying PCA and checking the varaince of individual components
def app_pca(sc):
    pca = PCA(n_components=3)
    pca.fit(sc[0])
    cum_sum = np.cumsum(pca.explained_variance_ratio_)
    return cum_sum
ap = app_pca(sc)



# Transforming the data based on 95% explained varaince
def pca(sc):
    npca = PCA(n_components=2)
    X_train_pca = npca.fit_transform(sc[0])
    X_test_pca  = npca.transform(sc[1])
    return X_train_pca ,X_test_pca
fpca = pca(sc)



#Predicting based on original data using the Logistic regression model
def log_regres(X_train_scaled, X_test_scaled,y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    acc_score = model.score(X_test_scaled,y_test)
    return round(acc_score,2)*100

log_reg =  log_regres(sc[0], sc[1], sp[1], sp[3])



#Predicting PCA
def pred_pca(X_train_pca,X_test_pca, y_train, y_test):
    mod_pca = LogisticRegression()
    mod_pca.fit(X_train_pca, y_train)
    acc_score = mod_pca.score(X_test_pca,y_test)
    return round(acc_score,2)*100
pca_pred = pred_pca(fpca[0], fpca[1], sp[1], sp[3])

print('Checking the varaince of individual components:\n', ap ,'\nSo I decided transforming the data based on 96% explained varaince' )
print('\nAccuracy score without PCA:',log_reg)
print('Accuracy score after applying PCA:',pca_pred)