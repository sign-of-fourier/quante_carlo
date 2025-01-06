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
import re
from datetime import datetime

from IPython.display import display, clear_output


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def post_history(compound_request):
    arguments = compound_request.split('|')
#    return requests.post(arguments[0], data=json.loads(arguments[1]))
    return requests.post(arguments[0], data=arguments[1])

class session:
    def __init__(self, model, hp_ranges, batch_sz, n_gpr_processors, n_processors, n_iter, 
            other_parameters, use_qc, bo_url, log_file):
        self.gpr_batch_size = batch_sz
        self.other_parameters = other_parameters
        self.n_iter = n_iter
        self.n_gpr_processors = n_gpr_processors
        self.model = model
        self.hp_ranges = hp_ranges
        self.hp_types = ['int' if type(x[0]) == int else 'float' for x in hp_ranges]
        self.n_processors = n_processors
        self.multivariate = True
        self.use_qc = use_qc
        self.bo_url = bo_url
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

    def initialize_gpr(self, p):
        
        self.next_points = self.get_random_points(self.n_processors)
        self.hp_history = self.next_points
        
        self.test_new_points(p)
        
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


    
    def test_new_points(self, p):
        
        parameters = []
        for i in range(self.n_processors):
            parameters.append(self.other_parameters.copy())
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
        #if p['use_qc'] == 'True':
        #    stem = 'https://boaz.onrender.com'
        #else:
        #    stem = 'http://localhost:8000'
        url = self.bo_url + "/bayes_opt?hp_types={}&g_batch_size={}&hp_ranges={}&y_best={}&n_gpus={}&use_qc={}".format(hp_types, 
             self.gpr_batch_size, 
             hp_ranges, 
             self.y_best,
             self.n_processors, 
             self.use_qc)
        #urls = [stem + str(i) + '&y_best='+str(self.y_best) for i in range(self.n_gpr_processors)]
        
        historical_points = ';'.join(self.history['points'])
        historical_scores = ','.join([str(s) for s in self.history['scores']])
        #return url+'|'+json.dumps({'scores': historical_scores, 'points': historical_points})
        self.worker_results = p.map(post_history, [url+'|'+json.dumps({'scores': historical_scores, 'points': historical_points})]*self.n_processors)
        
        jsponse = [json.loads(re.sub('inf', '10', a.content.decode('utf-8'))) for a in self.worker_results]
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
        self.initialize_gpr(p)
        iteration_id = [0] * self.n_processors
        for j in range(self.n_iter):


            self.log("iteration {}".format(j))
            start = time.time()
            self.get_new_points(p)
            seconds_getting_next_point = time.time()-start
            self.log("- {} seconds getting next points".format(round(seconds_getting_next_point, 2)))
            start = time.time()
            self.test_new_points(p)
            seconds_testing = time.time() - start
            self.log("- {} seconds testing next points".format(round(seconds_testing, 2)))
            iteration_id += [j+1]*self.n_processors
            self.log("- current best {}".format(self.y_best))
            clear_output(wait=True)
            print("\033[1m{}\033[0m out of \033[1m{}\033[0m ".format(j+1, self.n_iter))
            print("Seconds getting next points \033[91m{}\033[0m,  Seconds testing next points {} Current best accuracy: {}".format(round(seconds_getting_next_point, 4),
                                                                                                                        round(seconds_testing, 4), round(self.y_best, 4)))



            self.set_iteration_id(iteration_id)
#def session(f, limits, gpr_batch_size, n_gpr_processors, n_processors, n_iterations, other_parameters={}, log_file='/tmp/qclog_file.txt'):

#        return q
#    return qc_tune


