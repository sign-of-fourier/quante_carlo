from sklearn.neural_network import MLPRegressor
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

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



def r2(p, y_test):
    mu = np.mean(y_test)
    SSE = np.mean([(prediction-truth)**2 for prediction, truth in zip(p, y_test)])
    SST = np.mean([(prediction-mu)**2 for prediction, truth in zip(p, y_test)])
    return max(1-SSE/SST, -10)

def worker(x):
    return x*x-2


def nn(p):

    regr_sgd = MLPRegressor(random_state=1, max_iter=p['max_iter'], solver='sgd', 
                            hidden_layer_sizes = p['hidden_layer'],
                            alpha = p['alpha'],
                            learning_rate = 'adaptive',
                            momentum=p['momentum'], 
                            nesterovs_momentum = True).fit(p['X_train'], p['y_train'])

    score = r2(regr_sgd.predict(p['X_test']), p['y_test'])
    return [score, p['hidden_layer']]


#def qEI(new_points, y_best, q, mu, sigma, tail_recursion):
#    best_ccdf = 0
#    best_ids = []
#    for a in range(len(new_points)):
#        if a not in tail_recursion:
#            if q > 0:
#                ccdf, ids = qEI(new_points,  y_best, q-1, mu, sigma, tail_recursion + [a])
#                if ccdf > best_ccdf:
#                    best_ccdf = ccdf
#                    best_ids = ids
#            else:
#                cv = []
#                m = mu[tail_recursion + [a]]
#                for b in tail_recursion + [a]:
#                    cv.append(sigma[b][tail_recursion + [a]])
#                ccdf = 1
#                for d in range(len(m)):
#                    ccdf = ccdf * norm.cdf(y_best, m[d], cv[d][d])
#                ccdf = 1 - ccdf
#                if ccdf > best_ccdf:
#                    best_ccdf = ccdf
#                    best_ids = tail_recursion + [a]
#    return best_ccdf, best_ids


def qEI(p):
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


def qEI_wrapper(p):
    return qEI(p['new_points'], p['y_best'], p['q'], p['mu'], p['sigma'], [])




