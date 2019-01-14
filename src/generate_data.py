import os,random
from random import seed
from collections import defaultdict
import csv, sys, nltk
import numpy as np
from numpy import zeros
import itertools
import logging
import json

logging.basicConfig(level=logging.DEBUG)

def main():
    premise = []
    claim = []
    warrant = []

    #Premise,claim extraction
    warrant_0, warrant_1, label = [], [], []
    with open('/home/keshavsingh/train.tsv','r') as f:
        for line in csv.reader(f, delimiter='\t'):
            t_premise = line[4]
            t_claim = line[5]
            t_label = line[3]
            t_warrant_1 = line[2]
            t_warrant_0 = line[1]
            premise.append(t_premise)
            claim.append(t_claim)
            warrant_0.append(t_warrant_0)
            warrant_1.append(t_warrant_1)
            label.append(t_label)

    #Warrant correct extraction
    warrants_labels = list(zip(warrant_0,warrant_1,label))
    warrant_correct = []
    for w0,w1,l in warrants_labels:
        if l == '0':
            w = w0
        else:
            w = w1
        warrant_correct.append(w)

    #Generating data (positive and negative)
    pos_dataset = list(zip(premise, claim, warrant_correct))
    neg_dataset = []
    
    random.seed(123)
    
    for prem, claim, warrant in pos_dataset:
        premise_pool = []

        for p, c, w in pos_dataset:
            if c != claim and p != prem:
                premise_pool += [p]

        neg_claim = claim
        neg_premise = random.sample(premise_pool, 1)[0]
        neg_dataset += [(neg_premise, neg_claim, warrant)]

    # Create the dataset.
    inst = []
    
    for p, c, w in pos_dataset:
        print(json.dumps({
            "label": 1,
            "claim": c,
            "premise": p,
            "warrant": w,
        }))
        
    for p, c, w in neg_dataset:
        print(json.dumps({
            "label": 0,
            "claim": c,
            "premise": p,
            "warrant": w,
        }))

    
if "__main__" == __name__:
    main()