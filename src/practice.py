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
    with open(args.data) as f:
        data = [json.loads(line) for line in f]

    for i in data:
        premise_pool = []
        label_pool = []
        claim = i.get("claim")
        label = i.get("label")
        premise = i.get("premise")
        premise_pool += [premise]
        label_pool += [label]
        if label == 1:
            for j in data:
                c = j.get("claim")
                l = j.get("label")
                p = j.get("premise")
                if l != label and c == claim and p != premise:
                    premise_pool += [p]
                    label_pool += [l]
        else:
            break        
                    
        print(json.dumps({
            "claim": claim,
            "premise": premise_pool,
            "label": label_pool,
        }))

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data','--data', dest='data', required=True,
        help="Data to be parsed.")
    args = parser.parse_args()
    main(args)
