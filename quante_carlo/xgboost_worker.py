import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse

def instance(p):
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    bst = XGBRegressor(n_estimators=p['hparameters'][0], 
                       max_depth=p['hparameters'][1], 
                       learning_rate=p['hparameters'][2],
                       eval_metric='rmsle')

    bst.fit(X_train[:p['n_training_samples']], y_train[:p['n_training_samples']])

    preds = bst.predict(X_test)

    return 1-mse(preds, y_test)






