import numpy as np
import random
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor 
import pandas as pd
import warnings
import time

class hp_tuning_session:
    def __init__(self, model, layer_ranges, kernel, batch_sz, n_gpr_processors, n_processors):
        self.gpr_batch_size = batch_sz
        self.kernel = kernel
        self.n_gpr_processors = n_gpr_processors
        self.model = model
        self.layer_ranges = layer_ranges
        self.n_processors = n_processors
        self.multivariate = True
        self.qc = False
        self.score_history = []
        self.y_best = -1
        self.iteration_id = []
        self.ei_history = [-1]*n_processors


    def initialize_gpr(self, p, other_parameters):
        
        self.next_points = self.get_random_points(self.n_processors)
        self.layer_history = self.next_points
        
        self.test_new_points(p, other_parameters)
        
    def get_random_points(self, batch_size):
        
        points = []
        for i in range(batch_size):
            layers = []
            for r in self.layer_ranges:
                if type(r[0]) == int:
                    layers.append(random.randint(r[0], r[1]))
                else:
                    layers.append(random.random()*(r[1]-r[0])+r[0])
            points.append(tuple(layers))  
        return points




    def qei(self, batch_id):
        
        best_ids = []
        best_ccdf = 0
        
        for batch in self.batch_points[batch_id]:
            
            mu, sigma = self.gpr.predict(batch, return_cov=True)
        
            if self.multivariate:
            #    if self.qc:
            #        ccdf = 1 - qc([p['y_best']] * p['q'] - p['mu'][d], [p['sigma'][q][d] for q in d])
            #    else:
                ccdf = 1 - multivariate_normal.cdf([self.y_best]*self.n_processors, mu, sigma)
            #else:
            #ccdf = 1 - np.prod([norm.cdf(p['y_best'], p['mu'][d[q]], p['sigma'][d[q]][d[q]]) for q in range(p['q'])])
            if ccdf > best_ccdf:
                best_ccdf = ccdf
                best_ids = batch
                
        return best_ccdf, best_ids
    
    def test_new_points(self, p, other_parameters):
        
        parameters = []
        for i in range(self.n_processors):
            parameters.append(other_parameters.copy())
            parameters[i]['hparameters'] = self.next_points[i]
            parameters[i]['thread_id'] = i
        
        self.scores = p.map(self.model, parameters)
        
        self.score_history += self.scores
        self.y_best = max([np.mean(s) for s in self.score_history])
        
    def get_new_points(self, p):
        
        self.gpr = GaussianProcessRegressor(kernel=self.kernel,random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.gpr.fit(self.layer_history, [np.mean(x) for x in self.score_history])

        # optimize krig model
        self.batch_points = [[self.get_random_points(self.n_processors) for n in range(self.gpr_batch_size)] for g in range(self.n_gpr_processors)]
        batch = p.map(self.qei, range(self.n_gpr_processors))
        best_ccdf = -1
        for i, j in batch:
            if i > best_ccdf:
                best_ccdf = i
                best_ids = j

        #best_ccdf, best_ids = self.qei(gpr, p)
        self.ei_history += [best_ccdf]*self.n_processors
        self.next_points = best_ids
        self.layer_history += best_ids
  
    def set_iteration_id(self, iteration_history):
        self.iteration_id = iteration_history

    def summary(self):
        if len(self.iteration_id) != len(self.score_history):
            iteration_id = [-1] * len(self.score_history)
        else:
            iteration_id = self.iteration_id

        return pd.DataFrame({'iteration': iteration_id, 'score': [np.mean(s) for s in self.score_history],
                             'hparameters': self.layer_history, 'qEi': self.ei_history})
            


def carlo(f, limits, kernel, gpr_batch_size, n_gpr_processors, n_processors, n_iterations, other_parameters={}):
    def qc_tune_nn(p):

        q = hp_tuning_session(f, limits, kernel, gpr_batch_size, n_gpr_processors, n_processors)
        q.initialize_gpr(p, other_parameters)
        iteration_id = [0] * n_processors

        for j in range(n_iterations):
            print(j)
            start = time.time()
            q.get_new_points(p)
            print("  {} seconds getting next points".format(round(time.time()-start, 2)))
            start = time.time()
            q.test_new_points(p, other_parameters)
            print("  {} seconds testing next points".format(round(time.time()-start, 2)))
            iteration_id += [j+1]*n_processors

        q.set_iteration_id(iteration_id)
        return q
    return qc_tune_nn


