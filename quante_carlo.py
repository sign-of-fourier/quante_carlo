from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import time
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor 
import datetime as dt

def print_to_log(messages):
    
    if len(messages) == 1:
        info = messages[0] 
    else:
        info = "\n".join(["[{}] {}".format(dt.datetime.now(), m) for m in messages])

    print(info)

    
    with open('c:\\users\\markp\\onedrive\\desktop\\jupyter\\reliability\\qc_demo\\log.txt', 'a') as f:
        f.write("[{}] {}\n".format(dt.datetime.now(), info))


def qc(k, vcv):
    
    I = np.zeros((len(vcv), len(vcv)))
    for a in range(len(I)):
        I[a][a] = 1/np.sqrt(vcv[a][a])
    vcv = np.matmul(np.matmul(I, vcv), I)
    k = np.matmul(I, k)
    pt = np.prod(norm.cdf(k))
    pp = np.prod(norm.cdf(k))
    x = norm.pdf(k)/norm.cdf(k)
    spec = vcv
    for a in range(len(vcv)):
        vcv[a][a] = 0
    S = np.matmul(np.matmul(x, spec), x)*pp/2
    spec = [[cv**4 for cv in rho] for rho in spec]
    w = (norm.cdf(k) - k * norm.pdf(k))/norm.cdf(k)
    S1 = sum(np.matmul(spec, w)*pp)
    h = np.matmul(w, np.matmul(spec, w))*np.prod(norm.cdf(k))/2-S1+np.prod(norm.cdf(k))*sum([sum(s) for s in spec])/2

    g = pt + S
    return g + h


class parallel():
    def __init__(self, df, target_variable, fixed, hidden_layer_ranges, n_processors):
        print_to_log(['Initialize'])
        self.n_processors = n_processors
        self.fixed = fixed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df, target_variable, 
                                                                                test_size=0.33, random_state=42)
        self.train_time = []
        self.readable_layer_names = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
        self.n_layers = len(hidden_layer_ranges)
        self.p = Pool(processes = n_processors)
        self.kernel = DotProduct()+ WhiteKernel()
        self.krig_time = []
        self.performance = []
        self.iteration = []
        self.layers = []
        self.hidden_layer_ranges = hidden_layer_ranges
        print_to_log(["    size of search space: {}".format(np.prod([layer[1]-layer[0] for layer in hidden_layer_ranges]))])


    def set_initial_points(self, initial_layers):
        self.output = []
        for layer in initial_layers:
#            self.output.append(qei.nn_worker({'alpha': .001, 'momentum': .95, 'max_iter': self.nn_parameters['max_iter'], 
#                                              'hidden_layer': layer, 'X_train': self.X_train, 'y_train': self.y_train, 
#                                              'X_test': self.X_test, 'y_test': self.y_test}) )
            
            self.output.append(worker({'fixed': self.fixed, 'hidden_layer': layer, 'X_train': self.X_train, 'y_train': self.y_train,
                                           'X_test': self.X_test, 'y_test': self.y_test} ))

    def test_new_points(self):
 
 
        start = time.time()
        parameters = []
        for layer in self.new_points:
            parameters.append({'fixed': self.fixed, "X_train": self.X_train, 
                               "y_train": self.y_train, "X_test": self.X_test, "y_test": self.y_test, "hidden_layer": layer})
        self.output = self.p.map(worker,[i for i in parameters])

        self.train_time.append(time.time()-start)
    
        print_to_log(["    train time {}".format(round(time.time()-start,4))])
            
    def aggregate_results(self, itrx):    
        # add results of new points
        
        for o in self.output:
            # put 0 back in; kriging does use it
            print_to_log(["    r2: {}, points: {}".format(round(o[0], 3), o[1])])
            self.performance.append(o[0])
            self.iteration.append(itrx)
            self.layers.append(o[1])
        
        self.best_y = max(self.performance)
        self.BO = pd.DataFrame(self.layers)
        self.BO.columns = self.readable_layer_names[:self.n_layers]
        self.BO['performance'] = self.performance
        self.BO['iteration'] = self.iteration
#        print_to_log(self.BO.sort_values('performance', ascending=False).head().to_string().split("\n"))
        

        
    def get_next_points(self, BatchSize, use_qc):

        gpr = GaussianProcessRegressor(kernel=self.kernel,random_state=0)
        gpr.fit(self.layers, self.performance)

        # optimize krig model
        
        next_points = [tuple([np.random.randint(self.hidden_layer_ranges[layer_id][1]-self.hidden_layer_ranges[layer_id][0])
                              + self.hidden_layer_ranges[layer_id][0] for layer_id in range(self.n_layers)])
                       for t in range(BatchSize*self.n_processors)]
            
        mu, sigma = gpr.predict(next_points, return_cov=True)
        
        start = time.time()

        qEI_batches = [] 
        for n in range(self.n_processors):    
    
            # create a batch for qEI to search through
            # each row is a group from the "master pool of groupings"
            # sub subprocess will check all the groupings in the batch
                
            B = []
            for z in range(BatchSize):
                B.append(random.sample(range(len(next_points)), self.n_processors)) # might get some overlap
            qEI_batches.append({'new_points': B, 
                                'y_best': self.best_y, 'qc': use_qc,
                                'multivariate': True,
                                'q': self.n_processors, 'mu': mu, 'sigma': sigma}) 
    
        output2 = self.p.map(qei,[i for i in qEI_batches])
        self.krig_time.append(time.time()-start)
        print_to_log(["    optimize GP time {}".format(round(time.time()-start,4))])
            
        top_points = pd.DataFrame(output2)
        top_points.columns = ['score', 'points']
        self.new_points = [next_points[x] for x in top_points.sort_values('score', ascending=False)['points'][0]]




def qei(p):
    best_ids = []
    best_ccdf = 0
    for d in p['new_points']:
        if p['multivariate']:
            if p['qc']:
                ccdf = 1 - qc([p['y_best']] * p['q'] - p['mu'][d], [p['sigma'][q][d] for q in d])
            else:
                ccdf = 1 - multivariate_normal.cdf([p['y_best']] * p['q'], p['mu'][d], [p['sigma'][q][d] for q in d])
        else:
            ccdf = 1 - np.prod([norm.cdf(p['y_best'], p['mu'][d[q]], p['sigma'][d[q]][d[q]]) for q in range(p['q'])])
        if ccdf > best_ccdf:
            best_ccdf = ccdf
            best_ids = d
    return best_ccdf, best_ids


def r2(p, y_test):
    mu = np.mean(y_test)
    SSE = np.mean([(prediction-truth)**2 for prediction, truth in zip(p, y_test)])
    SST = np.mean([(prediction-mu)**2 for prediction, truth in zip(p, y_test)])
    return max(1-SSE/SST, -10)



def worker(p):
        
    regr_sgd = MLPRegressor(random_state=1, max_iter=p['fixed']['max_iter'], solver='sgd', 
                            hidden_layer_sizes = p['hidden_layer'],
                            alpha = p['fixed']['alpha'],
                            learning_rate = 'adaptive',
                            momentum=p['fixed']['momentum'], 
                            nesterovs_momentum = True).fit(p['X_train'], p['y_train'])

    score = r2(regr_sgd.predict(p['X_test']), p['y_test'])
    return [score, p['hidden_layer']]


def qc_tune_nn(batch_size_proc_gp, initial_layers, n_iterations, n_processors, next_points_method, tunable,  fixed, train, target):
    q = parallel(train, target, fixed, tunable['hidden_layer_ranges'], n_processors)
    q.set_initial_points(initial_layers)
        
    for itrx in range(n_iterations):
        print_to_log(["Iteration: " + str(itrx)])

        q.aggregate_results(itrx)
        if next_points_method == 'qc':
            q.get_next_points(batch_size_proc_gp, True)
        elif next_points_method == 'random':
            q.new_points = [tuple([np.random.randint(q.hidden_layer_ranges[layer_id][1]-q.hidden_layer_ranges[layer_id][0])
                                      + q.hidden_layer_ranges[layer_id][0] for layer_id in range(q.n_layers)])
                               for t in range(q.n_processors)]
        else:
            q.get_next_points(batch_size_proc_gp, False)

        q.test_new_points()
            

    return q.BO
