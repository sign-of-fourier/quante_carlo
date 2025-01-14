import requests
import json
import re

class BEO:
    def __init__(self, bo_batch_size=10, n_procs=4, use_qc=False,
                 qei_mpi=None,
                 url='https://boaz.onrender.com'):
        self.bo_batch_size = bo_batch_size
        self.bo_url = url
        self.qei_mpi = qei_mpi
        self.use_qc = use_qc
        self.n_processors = n_procs 

    def suggest_embeddings(self, description_embeddings, historical_ids, 
                           historical_scores):
        url = self.bo_url +\
        "/bayes_{}?g_batch_size={}&n_gpus={}&use_qc={}".\
        format(self.qei_mpi,
               self.bo_batch_size,
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