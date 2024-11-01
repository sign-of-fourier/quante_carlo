import numpy as np
import random
import json
import requests
from scipy.stats import multivariate_normal
#from sklearn.gaussian_process import GaussianProcessRegressor 
import pandas as pd
import warnings
import time
import os
from datetime import datetime


def post_history(compound_request):
    arguments = compound_request.split('|')
#    return requests.post(arguments[0], data=json.loads(arguments[1]))
    return requests.post(arguments[0], data=arguments[1])

class session:
    def __init__(self, model, hp_ranges, batch_sz, n_gpr_processors, n_processors, n_iter, other_parameters, log_file):
        self.gpr_batch_size = batch_sz
        self.other_parameters = other_parameters
        self.n_iter = n_iter
        self.n_gpr_processors = n_gpr_processors
        self.model = model
        self.hp_ranges = hp_ranges
        self.hp_types = ['int' if type(x[0]) == int else 'float' for x in hp_ranges]
        self.n_processors = n_processors
        self.multivariate = True
        self.qc = False
        self.score_history = []
        self.y_best = -1
        self.iteration_id = []
        self.ei_history = [-1]*n_processors
        self.logfile = log_file
        if not os.path.isfile(log_file):
            with open(log_file, 'w') as f:
                f.write("Starting\n")

    def log(self, x):

        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(self.logfile, 'a') as f:
            f.write('[' + dt + '] '+ x + "\n")

    def initialize_gpr(self, p, other_parameters):
        
        self.next_points = self.get_random_points(self.n_processors)
        self.hp_history = self.next_points
        
        self.test_new_points(p, other_parameters)
        
    def get_random_points(self, batch_size):
        
        points = []
        for i in range(batch_size):
            hp = []
            for r in self.hp_ranges:
                if type(r[0]) == int:
                    hp.append(random.randint(r[0], r[1]))
                else:
                    hp.append(random.random()*(r[1]-r[0])+r[0])
            points.append(tuple(hp))  
        return points


    
    def test_new_points(self, p, other_parameters):
        
        parameters = []
        for i in range(self.n_processors):
            parameters.append(other_parameters.copy())
            parameters[i]['hparameters'] = self.next_points[i]
            parameters[i]['thread_id'] = i
        
        self.scores = p.map(self.model, parameters)
        
        self.score_history += self.scores
        self.y_best = max([np.mean(s) for s in self.score_history])
        self.history = {'scores': self.score_history, 'points': [','.join([str(x) for x in h]) for h in self.hp_history]}
        pd.DataFrame(self.history).to_csv('history.txt', index=False)

    def get_new_points(self, p):
        """
        makes an API request to get next set of points using batch EI
        """
        hp_ranges = ';'.join([','.join([str(x) for x in s]) for s in self.hp_ranges])
        hp_types = ','.join(self.hp_types)
        #stem = 'http://localhost:8000/bayes_opt?hp_types='+hp_types+'&g_batch_size='+str(self.gpr_batch_size)+'&layer_ranges='+hp_ranges
        url = "https://boaz.onrender.com/bayes_opt?hp_types={}&g_batch_size={}&hp_ranges={}&y_best={}&n_gpus={}&use_qc=False".format(hp_types, 
             self.gpr_batch_size, 
             hp_ranges, 
             self.y_best,
             self.n_processors)
        #urls = [stem + str(i) + '&y_best='+str(self.y_best) for i in range(self.n_gpr_processors)]
        
        historical_points = ';'.join(self.history['points'])
        historical_scores = ','.join([str(s) for s in self.history['scores']])
        #return url+'|'+json.dumps({'scores': historical_scores, 'points': historical_points})
        worker_results = p.map(post_history, [url+'|'+json.dumps({'scores': historical_scores, 'points': historical_points})]*self.n_processors)
        jsponse = [json.loads(a.content.decode('utf-8')) for a in worker_results]
        best_score = -1
        for score, points in zip([j['best_ccdf'] for j in jsponse], [j['next_points'] for j in jsponse]):
            if score > best_score:
                best_score = score
                best_points = points
        self.next_points = [tuple([int(x) if t == 'int' else float(x) for x, t in zip(s.split(','), self.hp_types)]) for s in best_points.split(';')]
        self.hp_history += self.next_points
        self.ei_history += [best_score]*self.n_processors

        return 1
  
    def set_iteration_id(self, iteration_history):
        self.iteration_id = iteration_history

    def summary(self):
        if len(self.iteration_id) != len(self.score_history):
            iteration_id = [-1] * len(self.score_history)
        else:
            iteration_id = self.iteration_id

        return pd.DataFrame({'iteration': iteration_id, 'score': [np.mean(s) for s in self.score_history],
                             'hparameters': self.hp_history, 'qEi': self.ei_history})
            

    def tune(self, p):
        self.initialize_gpr(p, self.other_parameters)
        iteration_id = [0] * self.n_processors
        for j in range(self.n_iter):
            print(j)
            self.log("iteration {}".format(j))
            start = time.time()
            self.get_new_points(p)
            self.log("- {} seconds getting next points".format(round(time.time()-start, 2)))
            start = time.time()
            self.test_new_points(p, self.other_parameters)
            self.log("- {} seconds testing next points".format(round(time.time()-start, 2)))
            iteration_id += [j+1]*self.n_processors
            self.log("- current best {}".format(self.y_best))
            self.set_iteration_id(iteration_id)
#def session(f, limits, gpr_batch_size, n_gpr_processors, n_processors, n_iterations, other_parameters={}, log_file='/tmp/qclog_file.txt'):

#        return q
#    return qc_tune


