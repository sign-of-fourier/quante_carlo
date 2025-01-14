import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM#AutoModelForCausalLM
import os

torch.cuda.empty_cache()

llama_format = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = "Write a product description for a water resistant, French made gold watch with an alarm."
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device='cuda:0')
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
#from transformers import AutoTokenizer, LlamaForCausalLM#AutoModelForCausalLM
#model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

model.to('cuda')



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
              
    # prompt = ["Create an example of a movie review and an associated rating of 'Positve', 'Neutral', or 'Negative'.",
    #           "Pick a movie you know or make one up.", "Give your answer in a JSON format surrounded by backticks.",
    #           "Only provide a JSON with a review and a rating.",
    #           "Do not provide any other context, introduction or explanation.",
    #           "\n\n###Example 1###\n```\n{\"review\":",
    #           "\"The movie Jack and the Beanstalk was terrible. How could anyone believe that someone would sell their cow for some beans? So stupid!\",\n",
    #           "\"rating\": \"Negative\"}\n```\n",
    #           "\n\n###Example 2###\n```\n{\"review\":",
    #           "\"This movie was amazing! I love Johnnie Depp. The part where he eats a carrot was so funny!\",\n",
    #           "\"rating\": \"Positive\"}\n```\n\n"]
    
    

    input_ids = tokenizer(llama_format.format(" ".join(background), " ".join(prompt)), return_tensors="pt")
    input_ids.to('cuda:0')
    
    outputs = model.generate(**input_ids, max_length=4096, 
                    no_repeat_ngram_size=3,
                    num_return_sequences=3, 
                    do_sample=True,
                    top_k=50, top_p=.95, temperature=.9)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result


if __name__ == '__main__':


    parser = argparse.ArgumentParser("create_exapmple")
    parser.add_argument("-n", "--n_examples", help="Number of examples to create.", type=int)
    parser.add_argument("-s", "--sentiment", help="Number of examples to create.", type=str)
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

