from flask import Flask, request
import requests
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
#import numpy as np
import json
import pickle

gpr = None

app = Flask(__name__)

@app.route("/gpr", methods=['POST'])
def predict():

    
    reload = request.args.get('reload')
    gpr_path = request.args.get('gpr_path')
    if reload == 'True':
        with open(gpr_path, 'rb') as m:
            gpr = pickle.load(m)

    data = json.loads(request.data)
    batches = data['batches'].split('|')
        
    predictions = []
    covariances = []
    
    for batch in batches:
        p, s = gpr.predict([[float(c) for c in r.split(',')] for r in batch.split(';')], return_cov=True)
        #p = [[float(c) for c in r.split(',')] for r in batch.split(';')]
        predictions.append(','.join([str(x) for x in p]))
        covariances.append(';'.join([','.join([str(x) for x in r]) for r in s]))

    return {'k': ';'.join(predictions), 
            'sigma': '|'.join(covariances)}
