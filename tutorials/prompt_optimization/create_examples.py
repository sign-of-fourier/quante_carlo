import argparse
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

