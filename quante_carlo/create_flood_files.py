from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('downsample.csv')

predictors = [c for c in data.columns if c not in ['FloodProbability']]
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
train_drop_y_na = data[~(data['FloodProbability'].isna())]
imp_mean.fit(train_drop_y_na[predictors])
train_imputed = pd.DataFrame(imp_mean.transform(train_drop_y_na[predictors]))
train_imputed.columns = predictors
train_imputed.to_csv('imputed_training.csv')
train_imputed['FloodProbability'] = train_drop_y_na['FloodProbability']
#X_train, X_test, y_train, y_test = train_test_split(data[predictors], pd.get_dummies(data['label']), train_size=.66)

X_train, X_test, y_train, y_test = train_test_split(data[predictors], train_imputed['FloodProbability'], train_size=.66)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
