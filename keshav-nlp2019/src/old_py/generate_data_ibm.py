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
    claim = []
    premise = []

    #Premise,claim extraction
    label, warrant = [], []
    with open(args.data, 'r', encoding = 'ISO-8859-1') as f:
        for line in csv.reader(f, delimiter='\t'):
            t_claim = line[0]
            t_premise = line[1]
            t_warrant = line[2]
            premise.append(t_premise)
            claim.append(t_claim)
            warrant.append(t_warrant)
            
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
    pos_dataset = list(zip(premise, claim, warrant))
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
