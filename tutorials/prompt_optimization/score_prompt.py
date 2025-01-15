
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
    

    few_shot = ["\n\n#### FEW SHOT EXAMPLES ####\n\n"]+ ["### EXAMPLE {} ###\n**Movie Review**\n\n{}\n\n```{}```\n".format(r[0]+1, r[1], s) for r, s in zip(enumerate(reviews), sentiments)]
#    few_shot = ["\n"]
    
    prompt = ["####INSTRUCTIONS####\n\n",
              "Read the review below. Determine if the review is 'positive' or 'negative'.",
              "Provide a one word answer surrounded by three backticks with no other explanation, rationale, introduction or any other text or punctionation."]+\
              few_shot + ["\n**Movie Review**\n\n{}\n".format(new_review)]

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
            results.append(re.sub('`', '', re.sub("\n", '', sentiment)).lower())

        else:
            print("{} not found".format(tp[1]))
            
    return results