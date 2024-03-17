import argparse
from sklearn import svm, metrics, model_selection, linear_model, ensemble
import pprint as pt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def main():

    x_train = pd.read_csv("XTrain.csv").to_numpy()
    y_train = pd.read_csv("YTrain.csv").to_numpy()
    x_test = pd.read_csv("XTest.csv").to_numpy()
    y_test = pd.read_csv("YTest.csv").to_numpy()

    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    print(X.shape)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    pca = PCA(n_components=90)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    n_comp = X_pca.shape[1]
    print(n_comp)
    col_name = []
    for i in range(n_comp):
        txt = "component {index}".format(index = i+1)
        col_name.append(txt)

    X_df = pd.DataFrame(X_pca, columns=col_name)
    X_df.to_csv('X_pca.csv')
    Y_df = pd.DataFrame(Y, columns=['year'])
    Y_df.to_csv('Y_pca.csv')

    model = RandomForestClassifier(n_estimators=90)

    print('testing')

    xtrain, xtest, ytrain, ytest = train_test_split(X_pca, np.ravel(Y),
                     test_size=0.2,
                     shuffle=True)

    model.fit(xtrain, ytrain)
    prediction = model.predict(xtest)
    print(accuracy_score(ytest, prediction))
    print(np.square(np.subtract(ytest,prediction)).mean())
    #mean_squared_error(ytest, prediction) #130


if __name__ == "__main__":
    main()