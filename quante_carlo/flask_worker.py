from flask import Flask, request
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
import pandas as pd
import warnings
from scipy.stats import multivariate_normal
import random

def get_random_points(hp_ranges, batch_size):

    points = []
    for i in range(batch_size):
        layers = []
        for r in hp_ranges:
            if type(r[0]) == int:
                layers.append(random.randint(r[0], r[1]))
            else:
                layers.append(random.random()*(r[1]-r[0])+r[0])
        points.append(tuple(layers))
    return points


def qei(hp_ranges, g_batch_size, history_path, y_best, n_procs):

    kernel = DotProduct()+ WhiteKernel()

    best_ids = []
    best_ccdf = 0
    points = [get_random_points(hp_ranges, n_procs) for a in range(g_batch_size)]

    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
    history = pd.read_csv(history_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gpr.fit([tuple([float(n) for n in p.split(',')]) for p in history.points], [float(s) for s in history.scores])
        

    for batch in points:
        mu, sigma = gpr.predict(batch, return_cov=True)
        ccdf = 1 - multivariate_normal.cdf([y_best]*n_procs, mu, sigma)
        if ccdf > best_ccdf:
            best_ccdf = ccdf
            best_ids = batch
                
    return best_ccdf, best_ids


app = Flask(__name__)




@app.route("/bayes_opt")
def kriging():
   
   task_file = request.args.get('task_file') 
   y_best = request.args.get('y_best')
   gpr_batch_size = request.args.get('g_batch_size')
   layer_ranges = request.args.get('layer_ranges')
   hp_types = request.args.get('hp_types')
   n_gpus = request.args.get('n_gpus')
   hp_ranges_numerical = [x.split(',') for x in layer_ranges.split(';')]

   hp_ranges = [(int(x[0]), int(x[1])) if y=='int' else (float(x[0]), float(x[1])) for x, y in zip(hp_ranges_numerical, hp_types.split(','))]
   ccdf, best_ids = qei(hp_ranges, int(gpr_batch_size), 'history.txt', float(y_best), int(n_gpus))
   return "{\"next_points\": \""+';'.join([','.join([str(x) for x in s]) for s in best_ids])+"\",\"best_ccdf\": "+str(ccdf)+"}\n"





