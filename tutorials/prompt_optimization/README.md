Use Few Shot Optimization to improve prompts


LLMs can be used for basic machine learning tasks. Prompt optimization can be used as an alternative to or in combination with fine-tuning. Prompt optimization is less resource intensive and many times, out performs fine-tuning. In addition, prompt optimization can be easier to deploy since it does not require a separate fine-tuned model for each task, only separate prompts.

There are multiple ways to optimize a prompt. You can optimize the whole prompt or parts of the prompt. In this article we will focus on optimizing the “few shot” section of the prompt.

The dataset I will be working with is the IMDb dataset.

The task is to classify sentiment. Here is an example of a short prompt:

System
`
You are a sentiment analyzer.
Your job is to determine if a movie review is 'positive' or 'negative'.
User

####INSTRSUCTIONS####
Read the movie review below.
Determine if the movie review is 'positive' or 'negative'.
Surround your answer by backticks.

###EXAMPLE###
**Movie Review**
I loved this movie.
The part where Matt Dillon jumps on the trampoline was hilarious.
`positive`

**Movie Review**
This is the best movie I've come across in a long while. Not only is this the best movie of its kind(school shooting) 
The way Ben Coccio(the director) decided to film it was magnificent. He filmed it using teenage actors who were still attending high school.
He filmed it in the actors own rooms and used the actors real parents as their parents in the film. Also the actors were filming too using camcorders making it seem much more like a video diary. 
It is almost artful.(if that is indeed a word) There are a few slip ups however, for example when Cal calls brads(?) land rover a range rover(or vice versa, It's been awhile since I've seen it
Notice that there is an example. This is called 1 shot prompting. When there are a few examples, it is called few shot prompting. Few shot prompting has been known to improve the results from a LLM.
`

The optimization task is to select examples that improve the performance of the prompt. Since the data we have from Kaggle is labeled, we can measure the performance.

I will demonstrate Bayesian Optimization to select better examples:

Use the LLM to create examples. For simplicity, we will define this as the space of possible examples.
Convert the examples into embeddings. These embeddings will be used later in the Bayesian Optimization step.
Begin by selecting examples and inserting them into the prompt. For our demonstration, we will use one positive and one negative.
Run the prompt with the chosen examples for a set number of reviews from the dataset. For each review, the result of the prompt should be a label: ‘positive’ or ‘negative’.
Compare the results to the labels in the training set to generate a score such as F1 or accuracy.
Use the embeddings and the scores to perform Bayesian Optimization to generate a suggestion for which examples to try next.
The code included can be found at https://github.com/sign-of-fourier/quante_carlo/tree/main/tutorials

Training Set
First we create a file that contains paths to the reviews and the labels associated with those reviews.

ls aclImdb/train/pos/ | head -50 | awk '{print "positive,aclImbd/train/pos/" $1}' > train_paths.txt 
ls aclImdb/train/neg/ | head -50 | awk '{print "negative,aclImbd/train/neg/" $1}' >> train_paths.txt  
This will be the set used to generate the score.

Note: The three following scripts will all have use this library that loads the model and defines the prompt at the beginning. I’ve put it in library called llama.py

import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
import os

torch.cuda.empty_cache()

llama_format = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device='cuda:0')
model.to('cuda')
These scripts uses LlamaForCausalLM from the transformers library instead of in a Jupyter notebook. It uses a little less memory and it is easier to orchestrate as a .py file.

Notice the prompt structure for and the tokens for Llama. The pipeline from transformers normally handles this for you. Since we are using LlamaForCausalLM, we have to do this ourselves.

Create Examples
The first script to creates 500 positive examples and 500 negative examples. I’ve created a directory called ‘examples’ to store the created examples. Don’t get these confused with the labeled training dataset.


import llama

def create_example(pos_neg):

    background = ["You are movie critic teacher.",
                  "Your job is to teach people how to review movies using an example only.", "\n"]

    if pos_neg == 'positive':
        review = 'This movie was great! I love Johnnie Depp and Margo Robbie. The part where they jumped on the trampoline was awesome!'
    else:
        review = 'Terrible movie. Who would trade a cow for beans? Do not see Jack and the Beanie Stalk. The theater smelled and the popcorn was stale.'

    prompt = ["Write a {} movie review based on a movie that you know or make one up.".format(pos_neg),
              "Do not give any introductions, instructions, context, labels or explanations or any other text.", "Only provide a single movie review of 100 words or less.",
              "\n\n###Example###\n{}\n\n".format(review)]

    input_ids = llama.tokenizer(llama.template.format(" ".join(background), " ".join(prompt)), return_tensors="pt")
    input_ids.to('cuda:0')

    outputs = llama.model.generate(**input_ids, max_length=4096,
                                   no_repeat_ngram_size=3,
                                   num_return_sequences=3,
                                   do_sample=True,
                                   top_k=50, top_p=.95, temperature=.9)
    result = llama.tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser("create_exapmple")
    parser.add_argument("-n", "--n_examples", help="Number of examples to create.", type=int)
    parser.add_argument("-s", "--sentiment", help="Which sentiment to use: 'positve' or 'negative'.", type=str)
    parser.add_argument("-o", "--output_directory", help="Name of output directory.", type=str)
    args = parser.parse_args()

    results = []
    for x in range(args.n_examples):
        example = create_example(args.sentiment)

        assistant = False
        assistant_message = []
        for d in example.split("\n"):
            if assistant:
                assistant_message.append(d)
            if d == 'assistant':
                assistant = True
            results.append("\n".join(assistant_message))

            with open(args.output_directory + '/' + str(x) + '.txt', 'w') as f:
                f.write("\n".join(assistant_message))
This script uses a very basic prompt to create an example based on an example and label of ‘positive’ or ‘negative’. It is called with parameters that say where to store the results.

python create_examples.py -n 500 -s negative -o examples/negative
Next, we use Minish Lab embedding model to create embedding representations of the movie reviews. The snippet is not a script and can be run from a Notebook. The Minish Lab embedding encoders are very light weight.

from model2vec import StaticModel
import os

short_model = StaticModel.from_pretrained("minishlab/potion-base-2M")

# paths to selected examples, not using the whole dataset
positive_examples = os.listdir('examples/positive')
negative_examples = os.listdir('examples/negative')

# convert space to embeddings
for x in positive_examples:
    with open('examples/positive/' + x) as f:
        embedding= short_model.encode(f.read())
        
    with open('embeddings/positive/' + x, 'w') as f:
        f.write(','.join([str(x) for x in embedding]))

for x in negative_examples:
    with open('examples/negative/' + x) as f:
        embedding = short_model.encode(f.read())
        
    with open('embeddings/negative/' + x, 'w') as f:
        f.write(','.join([str(x) for x in embedding]))
Note: The next few scripts use the following three functions. The first one takes a list of examples, a list of their associated sentiments and a new review and then returns a result. The second one parses the output based on the Llama output format. The third one takes a pair of examples, their sentiments and goes through the training samples to produce an estimated label. I’ve put them in a library named score_prompt.py.


import os
import re
import llama


def extract_assistant(message):
        assistant = False
        assistant_message = []
        for d in message.split("\n"):
            if assistant:
                assistant_message.append(d)
            if d == 'assistant':
                assistant = True

        return "\n".join(assistant_message)
        
def evaluate(reviews, sentiments, new_review):
    
    background = ["You are a sentiment analyzer.",
                  "Your job is to read a review and determine if it is positve or negative."]

    few_shot = ["### EXAMPLE {} ###\n**Movie Review**\n\n{}\n\n```{}```\n".format(r[0]+1, r[1], s) for r, s in zip(enumerate(reviews), sentiments)]
    
    prompt = ["####INSTRUCTIONS####\n\n",
              "Read the review below. Determine if the review is 'positive' or 'negative'.",
              "Provide a one word answer surrounded by three backticks with no other explanation, rationale, introduction or any other text or punctionation.",
              "\n\n#### FEW SHOT EXAMPLES ####\n\n"] + few_shot + ["\n**Movie Review**\n\n{}\n".format(new_review)]

    input_ids = llama.tokenizer(llama.template.format(" ".join(background), " ".join(prompt)), return_tensors="pt")
    input_ids.to('cuda')
    
    outputs = llama.model.generate(**input_ids, max_length=4096, 
                        no_repeat_ngram_size=3,
                        num_return_sequences=3, 
                        do_sample=True,
                        top_k=50, top_p=.95, temperature=.01)
    result = llama.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return extract_assistant(result)


def predict(p):

    results = []
    for train_path in p['trainset_paths']:
        tp = train_path.split(",")  # be careful, tp has a different meaning
        if len(tp) == 1:
            continue
        if os.path.exists(tp[1]):
    
            with open(tp[1]) as r:
                train_review = r.read()
        
            sentiment = evaluate(p['reviews'], p['sentiments'], train_review)
            results.append(re.sub('`', '', re.sub("\n", '', sentiment)))

        else:
            print("{} not found".format(tp[1]))
            
    return results
2. Testing

This next script randomly selects two examples to embed in the prompt and then test on the labeled examples. This can be used to create the initial samples for the Bayesian Optimization process.

import argparse
import os
import score_prompt
import random
import re

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
    predicted_sentiments = []
    for train_path in trainset_paths:
        tp = train_path.split(",")
        if len(tp) == 1:
            continue
        if os.path.exists(tp[1]):
    
            with open(tp[1]) as r:
                train_review = r.read()
            
            sentiment = re.sub("\n", '', re.sub('`', '', score_prompt.evaluate(reviews, sentiments, train_review)))

            if not ((sentiment == 'positive') or (sentiment == 'negative')):
                sentiment = 'unreadable'
            predicted_sentiments.append(sentiment)
        else:
            print("{} not found".format(tp[1]))
            
    with open(args.output_filename, 'a') as f:
        f.write('|'.join(predicted_sentiments))
The script is called by passing the file of labels and review paths, the desired output file and the location of the positive and negative examples. I’ve created a directory for storing the scores named ‘scores’.

python rate_reviews.py -i train_paths.txt -o scores/1.txt -p examples/positive -n examples/negative
3. Parallel Bayesian Optimization

This next script also scores, but first, it takes the existing prompts that have been scored and performs Bayesian Optimization using a Gaussian Process Regressor. It then generates a batch of suggestions by optimizing batch Expected Improvement, a measure of how well a batch of suggestions balances the exploration, exploitation trade-off.

import argparse
import os
import random
import re
import pandas as pd
import score_prompt
import requests
import json
from scipy.stats import lognorm, multivariate_normal as mvn
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
            self.MUs = self.gpr.predict(self.candidates)

    def calculate_qei(self):
        
        if self.bo_batch_size == 1:
            self.scores = [np.exp(m+s**2/2)*(1-norm.cdf(self.y_best, m+s**2, s)) for m in self.MUs]
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
    qc.get_candidates(2000)
    results = qc.calculate_qei()

    sentiments = ['positive', 'negative']
    next_batch = pd.DataFrame({'ids': qc.candidate_ids, 'scores': qc.scores}).sort_values('scores', ascending=False)['ids'].iloc[0]
        
    if args.multiprocessing > 1:

        jobs = []

        for job_id, batch in enumerate(next_batch):

            traces, reviews = get_embeddings(id_to_pair[batch])

            jobs.append({'trainset_paths': trainset_paths, 'reviews': reviews.copy(), 'sentiments': sentiments, 'job_id': job_id})

        mp.set_start_method('spawn')
        p = mp.Pool(args.multiprocessing)
        job_results = p.map(score_prompt.worker, jobs)
        p.close()
        
        save(args.scored_directory, id_to_pair[batch], job_results)
        
    else:
        
        traces, reviews = get_embeddings(id_to_pair[next_batch])
        results = score_prompt.predict({'trainset_paths': trainset_paths, 'reviews': reviews.copy(), 'sentiments': sentiments})
        save(args.scored_directory, id_to_pair[next_batch], [results])

This script goes through the score cases and creates an input and an output for the Bayesian Process Regressor. The input to the objective function is the set of embeddings of the examples and the output is the F1 score.

Probability Integral Transform — The score is transformed so that it is a better fit for a lognormal. It is a better distribution for the Gaussian Process for computing Expected Improvement.

BO_Embedding — A class that creates a workspace for performing Bayesian Optimization on Embeddings. It has functions that will check what has been scored, train a Gaussian Process Regressor and then optimize that function to find the next best choice to evaluate

predict — This function performs the same loop that as in rate_reviews.py but does it in a separate function is a separate library taking a single json argument. This is easier for multiprocessing.

It is called in a similar way except that it includes the location of the directory where the scores are kept.

python optimize_few_shot.py -i train_paths.txt -p examples/positive -n examples/negative -s scores -m 2
Though the Bayesian Optimization can be run for a single example at a time, this script demonstrates batch Bayesian Optimization which suggests the optimal batch of examples to check. This script uses multiprocessing to orchestrate multiple worker processes. If you want to see how this works without multiprocessing you can simply pass 1 to the -m argument

Results
After 100 examples tested, here is the best pair of reviews to use as examples:

Postive

I just watched “Lost in Tokyo” and it’s a game-changer. The special effects are mind-blowing, the plot twists kept me on the edge of my seat, and the cast is phenomenal. The lead actress, Emma Stone, brings such depth to her chara
cter, and her chemistry with co-star Chris Evans is undeniable. The cinematography is stunning, capturing the neon-lit streets of Tokyo in breathtaking detail. I was completely absorbed in the world of the film, and I found myself
cheering for the underdog heroine until the very end. 10/10, would watch again!

Negative

Mind-numbingly dull. Why spend an entire film stuck in a world where characters are indistinguishable from cardboard cutouts? The plot was predictable and meandering, with a script that read like a laundry list of tired clichés. T
he acting was wooden and unconvincing, with dialogue that felt like it was being fed to me like a constant stream of bland corporate jargon. I checked my watch for the third time. Do yourself a favor and stay home.

F1:

No example: 78.7%
Manually create examples: 80.7%
Optimized Examples (100 evaluations): 86%
