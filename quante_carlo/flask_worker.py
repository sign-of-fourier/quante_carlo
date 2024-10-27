from flask import Flask, request
import requests
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
import pandas as pd
import warnings
from scipy.stats import multivariate_normal, norm, lognorm
import random
import numpy as np



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

def probability_integral_transform(scores):
    """
    It doesn't matter what space the scores are in during the Gaussian Process step. If they are similar 
    to a normal, they are only compared with the current best in that space
    """

    df = pd.DataFrame({'historical': scores, 'original_order': range(len(scores))})
    N = df.shape[0]
    df.reset_index(inplace=True)
    center_adjustment = 1/(2*N)
    df['uniform'] = [x/N+center_adjustment for x in df.index]
    df['normal'] = [np.log(p) for p in lognorm.ppf(df['uniform'], 1)]
    df.sort_values('original_order', inplace=True)
    return df['normal'].tolist()


def mvn_cdf(x, vcv):
    n = len(x)
    url = 'https://d9mj660cv0.execute-api.us-east-2.amazonaws.com/mvncdf/mvn'
    v_hat = np.sqrt(np.diagonal(vcv))
    x_hat = x/v_hat
    y = np.diag(1/v_hat)
    vcv_hat = np.matmul(y, np.matmul(vcv, y))
    # mvn.cdf(x/v_hat, [0, 0], np.matmul(y, np.matmul(vcv, y)))
    s_hat = ','.join([str(vcv_hat[a][b]) for a in range(0, n-1) for b in range(a+1, n)])
    x_hat = ','.join([str(x) for x in x_hat])
    response = requests.get(url + '?x=' + x_hat + '&s=' + s_hat)
    return float(response.text)


def qei(hp_ranges, g_batch_size, history, n_procs, use_qc):

    kernel = DotProduct()+ WhiteKernel()

    best_ids = []
    points = [get_random_points(hp_ranges, n_procs) for a in range(g_batch_size)]

    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
    # it really doesn't have to be a tuple
    historical_points = [tuple([float(p) for p in hpt.split(',')]) for hpt in history.get('points').split(';')]

    
    normal_scores = probability_integral_transform([float(s) for s in history.get('scores').split(',')])
    y_best = max(normal_scores)


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gpr.fit(historical_points, normal_scores)


    best_qei = -10
    for batch in points:
        try:
            mu, sigma = gpr.predict(batch, return_cov=True)
            for a in range(n_procs):
                sigma[a][a] = sigma[a][a]+.03
            _qei = 0
            for a in range(n_procs):

                mx = [[0]*n_procs for n in range(n_procs)]
                for b in range(n_procs):
                    for d in range(n_procs):
                        if (a != b) & (a != d):
                            mx[b][d] = sigma[b][d]-sigma[b][a]-sigma[d][a]+sigma[a][a]
                    mx[a][b] = -sigma[a][b]+sigma[a][a]
                    mx[b][a] = mx[a][b]

                mx[a][a]=sigma[a][a]
                m = [-y_best + sigma[a][a] + mu[m] if m == a else (mx[a][m] - mu[m]) for m in range(n_procs)]
            #print(sigma[a][a])
                try:
                    #print(m)
                    #print(mx)
                    #print(sigma[a][a])
                    if use_qc == 'True':
                        _qei += np.exp(sigma[a][a]/2)*mvn_cdf(m, mx)
                    else:
                        _qei += np.exp(sigma[a][a]/2)*multivariate_normal.cdf(m, [0]*n_procs, mx)
                    #_qei += multivariate_normal.cdf(m, [0]*n_procs, mx)
                except Exception as e:
                    with open('error_log.txt', 'a') as f:
                        f.write(e)
        except Exception as e:
            with open('other_error_log.txt', 'a') as f:
                f.write(e)
        if _qei > best_qei:
            best_qei = _qei
            best_ids = batch

    return best_qei, [i for i in best_ids]


def mpi(hp_ranges, g_batch_size, history_path, y_best, n_procs):

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




@app.route("/bayes_opt", methods=['GET', 'POST'])
def kriging():
   
   task_file = request.args.get('task_file') 
   gpr_batch_size = request.args.get('g_batch_size')
   hp_ranges = request.args.get('hp_ranges')
   hp_types = request.args.get('hp_types')
   n_gpus = request.args.get('n_gpus')
   use_qc = request.args.get('use_qc')

   data = request.form

   hp_ranges_numerical = [x.split(',') for x in hp_ranges.split(';')]
   hp_ranges = [(int(x[0]), int(x[1])) if y=='int' else (float(x[0]), float(x[1])) for x, y in zip(hp_ranges_numerical, hp_types.split(','))]
   ccdf, best_ids = qei(hp_ranges, int(gpr_batch_size), data,  int(n_gpus), use_qc)

   best_ids = [[int(g[i]) if t == 'int' else g[i] for i, t in enumerate(hp_types.split(','))] for g in best_ids]

   return "{\"next_points\": \""+';'.join([','.join([str(x) for x in s]) for s in best_ids])+"\",\"best_ccdf\": "+str(ccdf)+"}\n"





