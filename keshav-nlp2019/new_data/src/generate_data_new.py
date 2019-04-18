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
    evidence = []
    topic = []

    #Evidence, claim extraction
    label, warrant = [], []
    with open(args.data, 'r', encoding = 'ISO-8859-1') as f:
        for line in csv.reader(f, delimiter='\t'):
            t_label = line[3]
            t_claim = line[1]
            t_evidence = line[2]
            t_topic = line[0]
            if 'EXPERT' in t_label:
                evidence.append(t_evidence)
                claim.append(t_claim)
                topic.append(t_topic)

    pos_dataset = list(zip(evidence, claim, topic))
    neg_dataset = []

    random.seed(123)
    for evid, claim, topic in pos_dataset:
        true_premise_pool = []
        false_premise_pool = []
        for e, c, t in pos_dataset:
            if (t == topic and c == claim):
                true_premise_pool += [e]
            if (t == topic and c != claim):
                false_premise_pool += [e]

            false_premise_pool = [x for x in false_premise_pool if x not in true_premise_pool]

        neg_claim = claim
        if len(false_premise_pool) >= 2:
            neg_premise = random.sample(false_premise_pool, 1)[0]
            neg_dataset += [(neg_premise, neg_claim, topic)]
        
    # Create the dataset.
    for p, c, t in pos_dataset:
        print(json.dumps({
            "label": 1,
            "topic": t,
            "claim": c,
            "premise": p,
        }))

    for p, c, t in neg_dataset:
        print(json.dumps({
            "label": 0,
            "topic": t,
            "claim": c,
            "premise": p,
        }))


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data','--data', dest='data', required=True,
        help="Data to be parsed.")
    args = parser.parse_args()
    main(args)
