import requests
import json
import re

class bayesian_optimization_api:

    def __init__(self, batch_size, keys, hp_types, hp_space, n_procs, use_qc):
        self.batch_size = batch_size
        self.historical_parameters = ['.5,.5', '.4,.4']
        self.historical_values = [.5, 0.2]
        self.bo_url = 'https://boaz.onrender.com'
        self.keys = keys
        self.hp_types = [hp_types[k] for k in keys]
        self.hp_ranges = [hp_space[k] for k in keys]
        self.recent_update = True
        self.y_best = .5 # I don't think I even use this anymore
        self.use_qc = use_qc
        self.n_processors = n_procs
        self.local_buffer = []

    def register(self, parameters, value):
        self.historical_parameters.append(','.join([str(parameters[k]) for k in self.keys]))
        self.historical_values.append(value)
        self.recent_update = True

    def suggest(self):
        """
        makes an API request to get next set of points using batch EI
        """
        if self.recent_update or len(self.local_buffer) < 1:
            hp_ranges = ';'.join([','.join([str(x) for x in s]) for s in self.hp_ranges])
            hp_types = ','.join(self.hp_types)
            url = self.bo_url +\
                    "/bayes_opt?hp_types={}&g_batch_size={}&hp_ranges={}&y_best={}&n_gpus={}&use_qc={}".\
                    format(hp_types,
                           self.batch_size,
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
