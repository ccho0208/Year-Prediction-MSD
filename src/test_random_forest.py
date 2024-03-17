import sys, argparse
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


def main():

    # 1) load dataset
    x_train = pd.read_csv('XTrain.csv').to_numpy()
    y_train = pd.read_csv('YTrain.csv').to_numpy()
    x_test = pd.read_csv('XTest.csv').to_numpy()
    y_test = pd.read_csv('YTest.csv').to_numpy()

    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    print(X.shape)

    # 2) perform PCA
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    pca = PCA(n_components=90)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)


    n_comp = X_pca.shape[1]
    col_name = []
    for i in range(n_comp):
        txt = 'component {index}'.format(index = i+1)
        col_name.append(txt)

    X_df = pd.DataFrame(X_pca, columns=col_name)
    X_df.to_csv('X_pca.csv')
    Y_df = pd.DataFrame(Y, columns=['year'])
    Y_df.to_csv('Y_pca.csv')

    # 4) initiate random forest classifier
    model = RandomForestClassifier(n_estimators=50)

    xtrain, xtest, ytrain, ytest = train_test_split(X_pca, np.ravel(Y), test_size=0.2, shuffle=True)

    # 5) initiate random forest regression
    model = RandomForestRegressor(random_state=100, n_jobs=-1)

    params = {'n_estimators': [200],
              'max_depth': [2, 4, 5, 10, 15, 20, 25, 30],
              'max_features': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]}

    # 6) grid search
    grid_search = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, verbose=1, scoring='r2')
    grid_search.fit(xtrain, ytrain)

    model_best = grid_search.best_estimator_

    # 7) inference
    y_train_pred = model_best.predict(xtrain)
    y_test_pred = model_best.predict(xtest)

    print(r2_score(ytrain, y_train_pred))
    print(r2_score(ytest, y_test_pred))

    # 8) prediction
    prediction = model.predict(xtest)
    print(accuracy_score(ytest, prediction))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
