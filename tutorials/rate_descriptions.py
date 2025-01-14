import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


llama_format = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device='cuda:0')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model.to('cuda:0')

train_df = pd.read_csv('train.csv', encoding='iso-8859-1')


with open('descriptions4.txt') as f:
    descriptions = f.read().split('|')
    
def rate_description(background, search_term, description):
    prompt = "**Search Term** \n{}\n\n**Product Description**\n{}\n\n".format(search_term, description)

    input_ids = tokenizer(llama_format.format(" ".join(background), prompt), return_tensors="pt")
    input_ids.to('cuda:0')
    
    outputs = model.generate(**input_ids, max_length=1024, 
            no_repeat_ngram_size=3,
            num_return_sequences=3, 
            do_sample=True,
            top_k=50, top_p=.95, temperature=.05)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result
    
prompt2 = ["You are a search engine optimizer.", 
           "Your job is to evaluate the relevance of a product description.",
           "You will be given a search term and a product description.",
           "Rate the relevance of the product description on a scale between 0 and 3."]

    
train_df = pd.read_csv('train.csv', encoding='iso-8859-1')

for desc in descriptions:   
    score = [rate_description(prompt2, search, desc) for search in train_df[train_df['product_uid']==102456]['search_term']]

    with open('scores3.txt', 'a') as f:
        f.write('|'.join(score)+'||')
