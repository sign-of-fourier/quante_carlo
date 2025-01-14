import argparse
import os
import random

import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM#AutoModelForCausalLM

torch.cuda.empty_cache()




llama_format = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = "Write a product description for a water resistant, French made gold watch with an alarm."
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device='cuda:0')

model.to('cuda')



def evaluate(reviews, sentiments, new_review):
    
    background = ["You are a sentiment analyzer.",
                  "Your job is to read a review and determine if it is positve or negative."]
    

    few_shot = ["### EXAMPLE {} ###\n**Movie Review**\n\n{}\n\n```{}```\n".format(r[0]+1, r[1], s) for r, s in zip(enumerate(reviews), sentiments)]
    
    prompt = ["####INSTRUCTIONS####\n\n",
              "Read the review below. Determine if the review is 'positive' or 'negative'.",
              "Provide a one word answer surrounded by three backticks with no other explanation, rationale, introduction or any other text or punctionation.",
              "\n\n#### FEW SHOT EXAMPLES ####\n\n"] + few_shot + ["\n**Movie Review**\n\n{}\n".format(new_review)]

    input_ids = tokenizer(llama_format.format(" ".join(background), " ".join(prompt)), return_tensors="pt")
    input_ids.to('cuda:0')
    
    outputs = model.generate(**input_ids, max_length=4096, 
                    no_repeat_ngram_size=3,
                    num_return_sequences=3, 
                    do_sample=True,
                    top_k=50, top_p=.95, temperature=.01)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result
    
def extract_assistant(message):
        assistant = False
        assistant_message = []
        for d in message.split("\n"):
            if assistant:
                assistant_message.append(d)
            if d == 'assistant':
                assistant = True
        results.append("\n".join(assistant_message))
        
        return "\n".join(assistant_message)





import requests
import json
from scipy.stats import lognorm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF


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
    def __init__(self, url, kernel, 
                 embedding_space, bo_batch_size):
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
        
    def calculate_qei(self):
        url = self.url + '/qei?n={}&y_best={}'.format(self.bo_batch_size, self.y_best)
        S_as_str = []
        for s in self.COVs:
            S_as_str.append(';'.join([','.join([str(r) for r in row]) for row in s]))
        response = requests.post(url, data=json.dumps({'sigma': '|'.join(S_as_str),
                                                      'k': ';'.join([','.join([str(x) for x in row]) for row in self.MUs])}))
        try:
            jsponse = json.loads(response.content.decode('utf-8'))
            self.scores = [float(j) for j in jsponse['scores'].split(',')]
            return jsponse
        except Exception as e:
            print(e)
            return [-1]



if __name__ == '__main__':


    parser = argparse.ArgumentParser("rate_reviews")
    parser.add_argument("-i", "--input_filename", help="A file name containing paths that are a list of reviews.", type=str)
    parser.add_argument("-o", "--output_filename", help="Name of output directory.", type=str)
    parser.add_argument("-p", "--few_shot_positive", help="Name of directory of positive examples or path of single positive example.", type=str)
    parser.add_argument("-n", "--few_shot_negative", help="Name of directory of negative examples or path of single negative example.", type=str)
    
    args = parser.parse_args()
    with open(args.input_filename) as f:
        trainset_paths = f.read().split("\n")
        
    print("{} training samples".format(len(trainset_paths)))


    # if selecting 2 random reviews
    reviews = []
    traces = []

    if os.path.isdir(args.few_shot_positive) and os.path.isdir(args.few_shot_negative):
        for path, demos in zip([args.few_shot_positive, args.few_shot_negative],
                               [os.listdir(args.few_shot_positive), os.listdir(args.few_shot_negative)]):
            if path[-1] != '/':
                path = path + '/'
        
        
            reviews_id = path + random.choice(demos)
            traces.append(reviews_id)
            with open(reviews_id) as f:
                reviews.append(f.read())
                
    elif os.path.isfile(args.few_shot_positive) and os.path.isfile(args.few_shot_negative):
        for path in [args.few_shot_positive,
                     args.few_shot_negative]:
            with open(path) as f:
                reviews.append(f.read())
            traces.append(path)
    else:
        print('Directory path mismatch')
        exit(0)
        
        

    sentiments = ['positive', 'negative']

    with open(args.output_filename, 'w') as t:
        t.write(','.join(traces)+"\n")
    

    results = []
    for train_path in trainset_paths:
        tp = train_path.split(",")
        if len(tp) == 1:
            continue
        if os.path.exists(tp[1]):
    
            with open(tp[1]) as r:
                train_review = r.read()
            
            sentiment = evaluate(reviews, sentiments, train_review)
            with open(args.output_filename, 'a') as f:
                f.write(extract_assistant(sentiment)+'|')
        else:
            print("{} not found".format(tp[1]))
