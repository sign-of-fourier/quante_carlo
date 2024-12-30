import requests
import json
import re
import random

class BO_api:

    def __init__(self, bo_batch_size, keys, hp_types, hp_space, n_procs, use_qc):
        self.bo_batch_size = bo_batch_size
        self.historical_parameters = []
        self.historical_values = []
        self.bo_url = 'https://boaz.onrender.com'
        self.keys = keys
        self.hp_types = [hp_types[k] for k in keys]
        self.hp_ranges = [hp_space[k] for k in keys]
        self.recent_update = True
        self.y_best = .5 # I don't think I even use this anymore
        self.use_qc = use_qc
        self.n_processors = n_procs
        self.local_buffer = []
        self.min_parameter_history_size = 10

    def register(self, parameters, value):
        self.historical_parameters.append(','.join([str(parameters[k]) for k in self.keys]))
        self.historical_values.append(value)
        self.recent_update = True

    def suggest(self):
        """
        makes an API request to get next set of points using batch EI
        """
        if len(self.historical_parameters) < self.min_parameter_history_size:
            self.local_buffer = [','.join([str(random.randint(h[0], h[1])) if t == 'int' else str(random.uniform(h[0], h[1])) for h, t in zip(self.hp_ranges, self.hp_types)])]
        elif self.recent_update or len(self.local_buffer) < 1:
            hp_ranges = ';'.join([','.join([str(x) for x in s]) for s in self.hp_ranges])
            hp_types = ','.join(self.hp_types)
            url = self.bo_url +\
                    "/bayes_opt?hp_types={}&g_batch_size={}&hp_ranges={}&y_best={}&n_gpus={}&use_qc={}".\
                    format(hp_types,
                           self.bo_batch_size,
                           hp_ranges,
                           self.y_best,
                           self.n_processors,
                           self.use_qc)

            historical_points = ';'.join(self.historical_parameters)
            historical_scores = ','.join([str(s) for s in self.historical_values])

            response = requests.post(url, data=json.dumps({'scores': historical_scores, 'points': historical_points}))
            jsponse = json.loads(re.sub('inf', '10', response.content.decode('utf-8')))
            self.local_buffer = jsponse['next_points'].split(';')
            self.recent_update = False

        suggestion = self.local_buffer.pop()
        d = {}
        for k, t, v in zip(self.keys, self.hp_types, suggestion.split(',')):
            if t == 'int':
                d[k] = int(v)
            elif t == 'float':
                d[k] = float(v)

        return d


class Embeddings:

    def __init__(self, bo_batch_size=10, n_procs=4, use_qc=False,
                 url='https://boaz.onrender.com'):
        self.bo_batch_size = bo_batch_size
        self.bo_url = url
        self.use_qc = use_qc
        self.n_processors = n_procs 

    def suggestion_combinations(self, data):
        """
        Designed for case to just enumerate and go through many different combinations
        No tracking of 'historical' vs 'unevaluated'
        """
        
        url = self.bo_url +\
        "/embedding_mpi?g_batch_size={}&n_gpus={}&use_qc={}".\
        format(self.bo_batch_size,
               self.n_processors,
               self.use_qc)
               
        

        scores = ','.join([str(s) for s in data['scores']])
        response = requests.post(url, data=json.dumps({'scores': scores,
                                                       'embeddings': ';'.join([','.join([str(x) for x in e]) for e in data['embeddings']])}))
                                                       
        try:
            jsponse = json.loads(re.sub('inf', '10', response.content.decode('utf-8')))
        except Exception as e:
            print(e)
            print(response.content.decode('utf-8'))
            return [-1]
            
#        self.local_buffer = not_historical 
        return [int(x) for x in jsponse['next_points'].split(',')], jsponse['best_ccdf']
        
        



    def suggest_embeddings(self, description_embeddings, historical_ids, 
                           historical_scores):
        """
        Designed for case when there is 
        - a defined space
        - evaluated cases indicated by a list of ids
        - the rest are unevaluated cases
        suggests and evaluates
        """
        
        url = self.bo_url +\
        "/embedding_qei?g_batch_size={}&n_gpus={}&use_qc={}".\
        format(self.bo_batch_size,
               self.n_processors,
               self.use_qc)
               
        description_embeddings_str = []
        for desc in description_embeddings:
            description_embeddings_str.append(','.join([str(d) for d in desc]))
        
        evaluated_embeddings = [description_embeddings_str[e] for e in range(len(description_embeddings_str)) if e in historical_ids]
        unevaluated_embeddings = [description_embeddings_str[e] for e in range(len(description_embeddings_str)) if e not in historical_ids]
        not_historical = [e for e in range(len(description_embeddings)) if e not in historical_ids]

        
        scores = ','.join([str(s) for s in historical_scores])
        response = requests.post(url, data=json.dumps({'scores': scores, 'points': ';'.join(evaluated_embeddings),
                                                       'embeddings': ';'.join(unevaluated_embeddings)}))
        try:
            jsponse = json.loads(re.sub('inf', '10', response.content.decode('utf-8')))
        except Exception as e:
            print(e)
            print(response.content.decode('utf-8'))
            return [-1]
            
        self.local_buffer = not_historical 
        next_ids = [int(x) for x in jsponse['next_points'].split(',')]
        return [not_historical[n] for n in next_ids]
        
        
    def suggest_and_evaluate(self, worker, embedding_ids, totals, 
                             search_embeddings, description_embeddings, p):    
        
        suggestions = self.suggest_embeddings(description_embeddings, embedding_ids, totals)
        if suggestions[0] < 0:
            print("No suggestions")
            return [-1], [0] 
            
        suggested_embedding = [{'search_embeddings': search_embeddings, 
                               'description_embedding': description_embeddings[s].tolist()} for s in suggestions] 
        worker_results = p.map(worker, suggested_embedding)
        
        return suggestions, [sum(w) for w in worker_results]
        
        
        
        
