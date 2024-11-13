from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR

class prediction:
    def __init__(self, model, r):
        self.metric = 'r2'
        if model == 'ElasticNet':
            self.model = ElasticNet(alpha = r[0], l1_ratio=r[1])
        elif model == 'SVR':
            self.model = make_pipeline(StandardScaler(), SVR(C=r[0], epsilon=r[1]))
        elif model == 'XGBoostRegressor':
            self.model = XGBRegressor(gamma=r[0], reg_lambda=r[1], colsample_bytree=r[2], 
                                 max_depth=r[3], min_child_weight=r[4], learning_rate=r[5])
        else:
            print('No Model')

        
    def more_or_less(self, x, y):
        if self.metric == 'rmse':
            if x < y:
                return True
            else:
                return False
        else:
            if x > y:
                return True
            else:
                return False
    
    def fit(self, data):
        
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        return r2_score(y_test, self.model.predict(X_test))    

#import pandas as pd
#import numpy as np
#from xgboost import XGBRegressor
#from sklearn.metrics import mean_squared_error as mse

class bunch:
    def __init__(self, d):
        self.data = d['data']
        self.target = d['target']
        

def regression_worker(p):

    model = prediction(p['model'], p['hparameters'])
    return model.fit(p['data'])
    
