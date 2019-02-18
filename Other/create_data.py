import json
# Load benchmark.
arc_dataset = [json.loads(ln) for ln in open('data/complete_arc.jsonl')]
ibm_acl_dataset = [json.loads(ln) for ln in open('data/ibm_acl_positive.jsonl')]

# Divide them
arc_c = []
for sentence in arc_dataset:
    arc_c.append(sentence["claim"])

# Divide them
ibm_c = []
for sentence in ibm_acl_dataset:
    ibm_c.append(sentence["claim"])

# Import and download stopwords from NLTK.
#from nltk.corpus import stopwords
#from nltk import download
#download('stopwords')  # Download stopwords list.

# Remove stopwords.
#stop_words = stopwords.words('english')
#sentence_obama = [w for w in sentence_obama if w not in stop_words]
#sentence_president = [w for w in sentence_president if w not in stop_words]
#print(len(stop_words))

import os


for sentence in ibm_c:
    for sent1 in arc_c:
        print "\t".join([sentence, sent1])
