import csv
import numpy as np 
import pandas as pd
from  sklearn.discriminant_analysis import LinearDiscriminantAnalysis # lda library

from sklearn.model_selection import train_test_split



# putting the data into dataframe
def data_to_pandas(pth: csv, column: list) -> pd.DataFrame:
    data=pd.read_csv(pth, columns=column)
    return data

# splitting into train and test
def split_tt(X, y) -> tuple:
    X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0, test_size=0.2,stratify=y)
    return X_train, X_test, y_train, y_test



def train_model_lda(x_train, y_train) :
    lda=LinearDiscriminantAnalysis(n_components=2) # reducing the component of the to 2
    model=lda.fit(x_train, y_train)
    return model