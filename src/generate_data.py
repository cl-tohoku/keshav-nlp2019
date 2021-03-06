import os,random
from random import seed
from collections import defaultdict
import csv, sys, nltk
import numpy as np
from numpy import zeros
import itertools
import logging
import json
import argparse

logging.basicConfig(level=logging.DEBUG)

def main(args):
    premise = []
    claim = []

    #Premise,claim extraction
    warrant_0, warrant_1, label = [], [], []
    with open(args.data, 'r') as f:
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
        
    # creating claim and warrant dictionary
    # claim_warrant = list(zip(claim,warrant_correct))
    # dict_cw = defaultdict(list)
    # for c, w in claim_warrant:
    #    dict_cw[c].append(w)

    ########## Generating premise, claim and random warrant (positive dataset)
    #pos_data = list(zip(premise,claim))
    #new_data = []
    #random.seed(33)
    #for pre,cla in pos_data:
    #    for c, w in dict_cw.items():
    #        warrant_pool = []
    #        if c == cla:
    #            warrant_pool += w
    #            new_data += [(pre, cla, random.sample(warrant_pool, 1)[0])]
        
    #Generating data (positive and negative)
    #pos_dataset = new_data
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data','--data', dest='data', required=True,
        help="Data to be parsed.")
    args = parser.parse_args()
    main(args)
