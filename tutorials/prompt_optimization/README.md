# Use Few Shot Optimization to improve prompts
<center>
<img width=50% height=auto src='https://github.com/sign-of-fourier/quante_carlo/blob/dev/tutorials/prompt_optimization/androids.png'></img>
</center>

LLMs can be used for basic machine learning tasks. Prompt optimization can be used as an alternative to or in combination with fine-tuning. Prompt optimization is less resource intensive and many times, out performs fine-tuning. In addition, prompt optimization can be easier to deploy since it does not require a separate fine-tuned model for each task, only separate prompts.

There are multiple ways to optimize a prompt. You can optimize the whole prompt or parts of the prompt. In this article we will focus on optimizing the “few shot” section of the prompt.

The dataset is from the IMDb dataset.

The task is to classify sentiment. Here is an example of a short prompt:

system 

```
You are a sentiment analyzer.
Your job is to determine if a movie review is 'positive' or 'negative'.
User
```

user

````
####INSTRSUCTIONS####
Read the movie review below.
Determine if the movie review is 'positive' or 'negative'.
Surround your answer by backticks.

###EXAMPLE###
**Movie Review**
I loved this movie.
The part where Matt Dillon jumps on the trampoline was hilarious.

```positive```

**Movie Review**
This is the best movie I've come across in a long while. Not only is this the best movie of its kind(school shooting) 
The way Ben Coccio(the director) decided to film it was magnificent. He filmed it using teenage actors who were still attending high school.
He filmed it in the actors own rooms and used the actors real parents as their parents in the film. Also the actors were filming too using camcorders making it seem much more like a video diary. 
It is almost artful.(if that is indeed a word) There are a few slip ups however, for example when Cal calls brads(?) land rover a range rover(or vice versa, It's been awhile since I've seen it
Notice that there is an example. This is called 1 shot prompting. When there are a few examples, it is called few shot prompting. Few shot prompting has been known to improve the results from a LLM.
````


The optimization task is to select examples that improve the performance of the prompt. Since the data we have from Kaggle is labeled, we can measure the performance.

This tutorial uses Bayesian Optimization to select better examples:

1. Use the LLM to create examples. For simplicity, we will define this as the space of possible examples.
2. Convert the examples into embeddings. These embeddings will be used later in the Bayesian Optimization step.
3. Begin by selecting examples and inserting them into the prompt. For our demonstration, we will use one positive and one negative.
4. Run the prompt with the chosen examples for a set number of reviews from the dataset. For each review, the result of the prompt should be a label: ‘positive’ or ‘negative’.
5. Compare the results to the labels in the training set to generate a score such as F1 or accuracy.
6. Use the embeddings and the scores to perform Bayesian Optimization to generate a suggestion for which examples to try next.

### Training Set
First we create a file that contains paths to the reviews and the labels associated with those reviews.
```
ls aclImdb/train/pos/ | head -50 | awk '{print "positive,aclImbd/train/pos/" $1}' > train_paths.txt 
ls aclImdb/train/neg/ | head -50 | awk '{print "negative,aclImbd/train/neg/" $1}' >> train_paths.txt  
```
This will be the set used to generate the score.

Note: The three following scripts will all have use this library that loads the model and defines the prompt at the beginning. I’ve put it in library called llama.py
```
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
```
These scripts uses LlamaForCausalLM from the transformers library instead of in a Jupyter notebook. It uses a little less memory and it is easier to orchestrate as a .py file.

Notice the prompt structure for and the tokens for Llama. The pipeline from transformers normally handles this for you. Since we are using LlamaForCausalLM, we have to do this ourselves.

### Create Examples
The first script to creates 500 positive examples and 500 negative examples. I’ve created a directory called ‘examples’ to store the created examples. Don’t get these confused with the labeled training dataset.

This script uses a very basic prompt to create an example based on an example and label of ‘positive’ or ‘negative’. It is called with parameters that say where to store the results.
```
python create_examples.py -n 500 -s negative -o examples/negative
```
Next, we use Minish Lab embedding model to create embedding representations of the movie reviews. The snippet is not a script and can be run from a Notebook. The Minish Lab embedding encoders are very light weight.
```
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
```
Note: The next few scripts use the following three functions. 
They are in a library named score_prompt.py.

```
extract_assistant(message)
```
Parses the output based on the Llama output format.
```
evaluate(reviews, sentiments, new_reivew)
```
Takes a list of examples, a list of their associated sentiments and a new review and then returns a result. 

```
predict(p)
```
Takes a pair of examples, their sentiments and goes through the training samples to produce an estimated label. 



### Testing

This next script randomly selects two examples to embed in the prompt and then test on the labeled examples. This can be used to create the initial samples for the Bayesian Optimization process.
The script is called by passing the file of labels and review paths, the desired output file and the location of the positive and negative examples. I’ve created a directory for storing the scores named ‘scores’.
```
python rate_reviews.py -i train_paths.txt -o scores/1.txt -p examples/positive -n examples/negative
```
### Parallel Bayesian Optimization

This next script also scores, but first, it takes the existing prompts that have been scored and performs Bayesian Optimization using a Gaussian Process Regressor. It then generates a batch of suggestions by optimizing batch Expected Improvement, a measure of how well a batch of suggestions balances the exploration, exploitation trade-off.

This script goes through the score cases and creates an input and an output for the Bayesian Process Regressor. The input to the objective function is the set of embeddings of the examples and the output is the F1 score.

Probability Integral Transform — The score is transformed so that it is a better fit for a lognormal. It is a better distribution for the Gaussian Process for computing Expected Improvement.

BO_Embedding — A class that creates a workspace for performing Bayesian Optimization on Embeddings. It has functions that will check what has been scored, train a Gaussian Process Regressor and then optimize that function to find the next best choice to evaluate

predict — This function performs the same loop that as in rate_reviews.py but does it in a separate function is a separate library taking a single json argument. This is easier for multiprocessing.

It is called in a similar way except that it includes the location of the directory where the scores are kept.
```
python optimize_few_shot.py -i train_paths.txt -p examples/positive -n examples/negative -s scores -m 2
```
Though the Bayesian Optimization can be run for a single example at a time, this script demonstrates batch Bayesian Optimization which suggests the optimal batch of examples to check. This script uses multiprocessing to orchestrate multiple worker processes. If you want to see how this works without multiprocessing you can simply pass 1 to the -m argument

### Results
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
