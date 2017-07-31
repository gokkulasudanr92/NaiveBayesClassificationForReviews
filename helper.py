import pandas as pd
import re
import math
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

def extract_data_list_from_csv(df, key):
    comments = []
    for index, row in df.iterrows():
        value = row[key]
        if (math.isnan(value)):
            break
        else:
            if value < 1.0:
	        feedback = 'neg'
	    else:
		feedback = 'pos'
        
        line = row["Comments"]
        line = re.sub("[^a-zA-Z?!]"," ", str(line))  #remove it
    	#words = [w.lower() for w in line.strip().split() if len(w)>=3]        #change it to cleanData
    	comments.append((line, feedback))

    return comments;


def extract_train_data(comments):
    return comments[:len(comments) - 200]

def extract_test_data(comments):
    return comments[len(comments) - 200: len(comments)]

def print_accuracy(text, model, test):
    print("For " + text + ":")
    print(text + " Accuracy: {0}".format(model.accuracy(test)))
    print
