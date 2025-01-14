import argparse
import os
import random
import re
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
            scores = ''.join(model_output[1:]).split('|')
        else:
            scores = model_output[1].split('|')
                
        sentiments = []
        sentiments = [re.sub("`", "", re.sub("\n", '', s)) for s in scores]
    
        tp = 0
        tn = 0

        for t, s in zip(truth, sentiments):
            if (t.split(',')[0] == 'negative') and (s == 'negative'):
                tn += 1
            
            if (t.split(',')[0] == 'positive') and (s == 'positive'):
                tp += 1
    
        fp_fn = len(scores) - tn - tp
        print("{} Accuracy {}, F1 {}".format(score_file,  (tp+tn)/len(scores), tp/(tp+.5*fp_fn)))
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
        
            sentiment = evaluate(reviews, sentiments, train_review)
            results.append(extract_assistant(sentiment))

        else:
            print("{} not found".format(tp[1]))
    return results

if __name__ == '__main__':


    parser = argparse.ArgumentParser("rate_reviews")
    parser.add_argument("-i", "--input_filename", help="A file name containing a list of paths to reviews.", type=str)
    parser.add_argument("-p", "--positive_examples", help="A file name containing paths that are a list of reviews.", type=str)
    parser.add_argument("-n", "--negative_examples", help="A file name containing paths that are a list of reviews.", type=str)
    parser.add_argument("-s", "--scored_directory", help="A file name containing paths that are a list of reviews.", type=str)

    args = parser.parse_args()
    with open(args.input_filename) as f:
        trainset_paths = f.read().split("\n")
        
    print("{} training samples".format(len(trainset_paths)))


    scored_embeddings, F1, scored_embedding_id_pairs = get_scored_emebeddings(args.input_filename, args.scored_directory)

    positive_examples = os.listdir(args.positive_examples)
    negative_examples = os.listdir(args.negative_examples)
    embedding_space, evaluated_ids, id_to_pair = get_embedding_space(positive_examples, negative_examples, scored_embedding_id_pairs) # a pairwise concatenation of one poitive and one negative
    

    qc = bo_embedding('http://localhost:8000', DotProduct()+WhiteKernel(), embedding_space, 4)
    qc.fit_gpr(scored_embeddings, F1)
    qc.get_candidates(2000)
    results = qc.calculate_qei()
    next_batch = pd.DataFrame({'ids': qc.candidate_ids, 'scores': qc.scores}).sort_values('scores', ascending=False)['ids'].iloc[0]

    for batch in next_batch:

        reviews = []
        traces = []
        
        for path in id_to_pair[batch].split(','):
            with open(re.sub('examples', 'embeddings', path)) as f:
                reviews.append(f.read())
            traces.append(path)

        

        sentiments = ['positive', 'negative']

        already_scored = os.listdir(args.scored_directory)
        i = 1
        
        score_filename = str(len(already_scored) + i) + '.txt' 

        while score_filename in already_scored:
            i += 1
            score_filename = str(len(already_scored) + i) + '.txt' 
            
        print(score_filename)
        results = score_examples(trainset_paths, reviews, sentiments)
        with open(args.scored_directory + '/' + score_filename, 'w') as f:
            f.write(id_to_pair[batch] + "\n" + '|'.join(results))
            
            