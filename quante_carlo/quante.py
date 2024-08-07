import numpy as np
import random
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor 
import pandas as pd
import warnings

class hp_tuning_session:
    def __init__(self, model, layer_ranges, kernel, n_batches, n_processors):
        self.n_batches = n_batches
        self.kernel = kernel
        self.model = model
        self.layer_ranges = layer_ranges
        self.n_processors = n_processors
        self.multivariate = True
        self.qc = False
        self.score_history = []
        self.y_best = -1
        self.iteration_id = []
        self.ei_history = [-1]*n_processors


    def initialize_gpr(self, p, keep_thread_id=False):
        
        self.next_points = self.get_random_points(self.n_processors)
        self.layer_history = self.next_points
        
        self.test_new_points(p, keep_thread_id)
        
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


    def qei(self, gpr):
        
        best_ids = []
        best_ccdf = 0
        for batch in self.batch_points:
            
            mu, sigma = gpr.predict(batch, return_cov=True)
        
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
    
    def test_new_points(self, p, pass_thread_id=True):

        if pass_thread_id:
            self.scores = p.map(self.model, [{'next_points': self.next_points[i],                                                                              'thread_id': i} for i in range(self.n_processors)])
        else:
            self.scores = p.map(self.model, self.next_points)
        
        self.score_history += self.scores
        self.y_best = max([np.mean(s) for s in self.score_history])
        
    def get_new_points(self):
        
        gpr = GaussianProcessRegressor(kernel=self.kernel,random_state=0)
        with warnings.catch_warnings(action='ignore'):
            gpr.fit(self.layer_history, [np.mean(x) for x in self.score_history])

        # optimize krig model
        self.batch_points = [self.get_random_points(self.n_processors) for n in range(self.n_batches)]
        best_ccdf, best_ids = self.qei(gpr)
        self.ei_history += [best_ccdf]*self.n_processors
        self.next_points = best_ids
        self.layer_history += best_ids

    def summary(self):
        return pd.DataFrame({'score': [np.mean(s) for s in self.score_history],
                             'layers': self.layer_history, 'ei': self.ei_history})
            


def carlo(f, limits, kernel, n_batches, n_processors, n_iterations, keep_thread_id=False):
    def qc_tune_nn(p):

        q = hp_tuning_session(f, limits, kernel, n_batches, n_processors)
        q.initialize_gpr(p, keep_thread_id)
        iteration_id = [0] * n_processors

#        for j in range(n_iterations):
#            q.get_new_points()
#            q.test_new_points(p, keep_thread_id)
#            iteration_id += [j+1]*n_processors

        return q#, iteration_id
    return qc_tune_nn


