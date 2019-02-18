import csv, os, random
from random import seed

from collections import namedtuple

#arc_dataset = namedtuple("arc_dataset", "id w0 w1 label premise claim topic description")

warrant_0, warrant_1, label, premise, claim = [], [], [], [], []
with open('complete_arcdata.txt', 'r', encoding = 'ISO-8859-1') as f:
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

pos_dataset = list(zip(premise, claim, warrant_correct))

random.seed(123)

new_dataset = []
with open('matched_claim_ibm_emnlp_test.txt', 'r') as f:
    for line in f:
        line = line.strip()
        premise_pool = []
        for p, c, w in pos_dataset:
            p = p.strip()
            c = c.strip()
            w = w.strip()
            if c.lower().split() == line.lower().split():
                premise_pool += [p]
#            else:
#                word = "toughened"
#                if word in line.lower().split() and word in c.lower().split():
#                    print(c.lower().split())
#                    print(line.lower().split())
#        print(premise_pool)
#        print(line)

        rand_premise = random.sample(premise_pool, 1)[0]
        new_dataset += [(line, rand_premise)]
fi = open('matched_premise_ibm_emnlp_test.txt', 'w')
for entry in new_dataset:
    fi.write(entry[1] + '\n')

fi.close()
