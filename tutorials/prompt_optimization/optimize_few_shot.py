import argparse
import os
import random
import re
import time
import pandas as pd
import score_prompt
import requests
import json
from scipy.stats import lognorm, norm, multivariate_normal as mvn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF
import multiprocessing as mp

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


class bo_embedding:
    """
    Requires that a defined search space be passed
    """
    def __init__(self, kernel, 
                 embedding_space, bo_batch_size, url='https://boaz.onrender.com'):
        self.url = url
        self.bo_batch_size = bo_batch_size
        self.embedding_space = embedding_space
        self.kernel = kernel

    def fit_gpr(self, embeddings, scores):

        normal_scores = probability_integral_transform(scores)
        self.y_best = max(normal_scores)

        gpr = GaussianProcessRegressor(kernel=self.kernel)
        self.gpr = gpr.fit(embeddings, normal_scores)
    
    def get_candidates(self, N):
        
        self.candidates = []
        self.candidate_ids = []
        self.MUs = []
        self.COVs = []
        for b in range(N):
            if self.bo_batch_size == 1:
                cid = random.randint(0, len(self.embedding_space)-1)
                self.candidate_ids.append(cid)
                self.candidates.append(self.embedding_space[cid])
            else:
                candidate = []
                candidate_id = []
                for a in range(self.bo_batch_size):
                    cid = random.randint(0, len(self.embedding_space)-1)
                    candidate.append( self.embedding_space[cid])
                    candidate_id.append(cid)
                self.candidate_ids.append(candidate_id.copy())
                self.candidates.append(candidate.copy())
                
                mu, sigma = self.gpr.predict(candidate, return_cov=True)
                self.MUs.append(mu.copy())
                self.COVs.append(sigma.copy())
                
        if self.bo_batch_size == 1:
            self.MUs, self.COVs = self.gpr.predict(self.candidates, return_std=True)

    def calculate_qei(self):
        
        if self.bo_batch_size == 1:
            self.scores = [np.exp(m+s**2/2)*(1-norm.cdf(self.y_best, m+s**2, s)) for m, s in zip(self.MUs, self.COVs)]
        else:
            url = self.url + '/qei?n={}&y_best={}'.format(self.bo_batch_size, self.y_best)
            S_as_str = []
            for s in self.COVs:
                S_as_str.append(';'.join([','.join([str(r) for r in row]) for row in s]))
            response = requests.post(url, data=json.dumps({'sigma': '|'.join(S_as_str),
                                                      'k': ';'.join([','.join([str(self.y_best-x) for x in row]) for row in self.MUs])}))
            try:
                jsponse = json.loads(response.content.decode('utf-8'))
                self.scores = [float(j) for j in jsponse['scores'].split(',')]
                return jsponse
            except Exception as e:
                print(e)
                print(response.content.decode('utf-8'))
                return [-1]


def get_scored_emebeddings(training_index, scores_directory):

    with open(training_index) as f:
        truth = f.read().split("\n")
    embeddings = []
    F1 = []
    paired_embedding_ids = []
    scores = os.listdir(scores_directory)
    for score_file in scores:
        with open(scores_directory + '/' + score_file) as f:
            model_output = f.read().split("\n") # first line is which samples are used
        
        if len(model_output) > 2:
            sentiments = ''.join(model_output[1:]).split('|')
        else:
            sentiments = model_output[1].split('|')
                
        tp = 0
        tn = 0
        u = 0
        for t, s in zip(truth, sentiments):
            true_sentiment = t.split(',')[0]
            
            if (true_sentiment == 'negative') and (s == 'negative'):
                tn += 1
            
            if (true_sentiment == 'positive') and (s == 'positive'):
                tp += 1
                
            if (true_sentiment == 'unreadable'):
                u += 1
        N = len(sentiments) - u
        
        fp_fn = N - tn - tp
        print("{} Accuracy {}, F1 {}".format(score_file,  (tp+tn)/N, tp/(tp+.5*fp_fn)))
        F1.append(tp/(tp+.5*fp_fn))
        paths = model_output[0].split(',')
        with open(re.sub('examples', 'embeddings', paths[0])) as f:
            pos_embedding = f.read()    
        with open(re.sub('examples', 'embeddings', paths[1])) as f:
            neg_embedding = f.read()
        
        paired_embedding_ids.append(model_output[0])
        
        e = pos_embedding + ',' + neg_embedding
        embeddings.append([float(x) for x in e.split(',')])
    
    return embeddings, F1, paired_embedding_ids

def get_embedding_space(postive_examples, negative_examples, paired_embedding_ids):
    
    
    positive_embeddings = []
    for x in positive_examples:
        with open('embeddings/positive/' + x) as f:
            positive_embeddings.append(','.join([str(round(float(x), 4)) for x in f.read().split(',')]))

    negative_embeddings = []
    for x in negative_examples:
        with open('embeddings/negative/' + x) as f:
            negative_embeddings.append(','.join([str(round(float(x), 4)) for x in f.read().split(',')]))


    # the embedding to optimize is just the positive embedding followed by the negative embedding
    embedding_space = []
    evaluated_ids = []
    id_to_pair = []
    ct = 0
    for pos, pname in zip(positive_embeddings, positive_examples):
        for neg, nname in zip(negative_embeddings, negative_examples):
            combined_embedding = pos + ',' + neg
            embedding_space.append([float(f) for f in combined_embedding.split(',')])
            pair = 'examples/positive/{},examples/negative/{}'.format(pname, nname)
            id_to_pair.append(pair)
            if pair in paired_embedding_ids:
                evaluated_ids.append(ct)
            ct += 1
            
    return embedding_space, evaluated_ids, id_to_pair



def score_examples(trainset_paths, reviews, entiments):
    
    results = []
    for train_path in trainset_paths:
        tp = train_path.split(",")  # be careful, tp has a different meaning
        if len(tp) == 1:
            continue
        if os.path.exists(tp[1]):
    
            with open(tp[1]) as r:
                train_review = r.read()
        
            sentiment = score_prompt.evaluate(reviews, sentiments, train_review)
            results.append(sentiment)

        else:
            print("{} not found".format(tp[1]))
    return results


def save(scored_directory, path_pair_string, job_results):
    
    for job in job_results:
        already_scored = os.listdir(scored_directory)

        i = 1
        score_filename = str(len(already_scored) + i) + '.txt' 
        while score_filename in already_scored:
            i += 1
            score_filename = str(len(already_scored) + i) + '.txt' 
            
        print(score_filename)    
        with open(scored_directory + '/' + score_filename, 'w') as f:
            f.write(path_pair_string + "\n" + '|'.join(job))
                

def get_embeddings(path_pair_string):
    reviews = []
    traces = []
    for path in path_pair_string.split(','):
        with open(re.sub('examples', 'embeddings', path)) as f:
            reviews.append(f.read())
        traces.append(path)
        
    return traces, reviews
    
    
    
def get_reviews(path_pair_string):
    reviews = []
    traces = []
    for path in path_pair_string.split(','):
        with open(path) as f:
            reviews.append(f.read())
        traces.append(path)
        
    return traces, reviews

if __name__ == '__main__':


    parser = argparse.ArgumentParser("rate_reviews")
    parser.add_argument("-i", "--input_filename", help="A file name containing a list of paths to reviews.", type=str)
    parser.add_argument("-p", "--positive_examples", help="A directory of positive examples.", type=str)
    parser.add_argument("-n", "--negative_examples", help="A directory or negative examples.", type=str)
    parser.add_argument("-m", "--multiprocessing", help="Number of parallel processes.", type=int)
    parser.add_argument("-s", "--scored_directory", help="A file name containing paths that are a list of reviews.", type=str)

    args = parser.parse_args()
    with open(args.input_filename) as f:
        trainset_paths = f.read().split("\n")
        
    print("{} training samples".format(len(trainset_paths)))


    scored_embeddings, F1, scored_embedding_id_pairs = get_scored_emebeddings(args.input_filename, args.scored_directory)

    positive_examples = os.listdir(args.positive_examples)
    negative_examples = os.listdir(args.negative_examples)
    embedding_space, evaluated_ids, id_to_pair = get_embedding_space(positive_examples, negative_examples, scored_embedding_id_pairs) # a pairwise concatenation of one poitive and one negative
    

    qc = bo_embedding(DotProduct()+WhiteKernel(), embedding_space, args.multiprocessing)
    qc.fit_gpr(scored_embeddings, F1)
    qc.get_candidates(5000)
    
    results = qc.calculate_qei()
    sentiments = ['positive', 'negative']
    next_batch = pd.DataFrame({'ids': qc.candidate_ids, 'scores': qc.scores}).sort_values('scores', ascending=False)['ids'].iloc[0]
        
    if args.multiprocessing > 1:

        jobs = []

        for job_id, batch in enumerate(next_batch):

            reviews = []
            for path in id_to_pair[batch].split(','):
                with open(path) as f:
                    reviews.append(f.read())
        
        #traces, reviews = get_embeddings()

            jobs.append({'trainset_paths': trainset_paths, 'reviews': reviews, 'sentiments': sentiments, 'job_id': job_id})

        #mp.set_start_method('spawn')
        #p = mp.Pool(args.multiprocessing)
        #job_results = p.map(score_prompt.worker, jobs)
        #p.close()
        
        job_results = []
        for j in jobs:
            job_results.append(score_prompt.predict(j))
        

        save(args.scored_directory, id_to_pair[batch], job_results)
        
    else:
        
        reviews = []
        for path in id_to_pair[next_batch].split(','):
            with open(path) as f:
                reviews.append(f.read())
        results = score_prompt.predict({'trainset_paths': trainset_paths, 'reviews': reviews, 'sentiments': sentiments})
        save(args.scored_directory, id_to_pair[next_batch], [results])

