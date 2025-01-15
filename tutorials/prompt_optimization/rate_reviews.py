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
            
            sentiment = re.sub("\n", '', re.sub('`', '', score_prompt.evaluate(reviews, sentiments, train_review))).lower()
            if not ((sentiment == 'positive') or (sentiment == 'negative')):
                sentiment = 'unreadable'
            predicted_sentiments.append(sentiment)
        else:
            print("{} not found".format(tp[1]))
            
    with open(args.output_filename, 'a') as f:
        f.write('|'.join(predicted_sentiments))
