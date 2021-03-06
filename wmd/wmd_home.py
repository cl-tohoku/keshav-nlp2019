import json
# Load benchmark.
arc_dataset = [json.loads(ln) for ln in open('data/complete_arc.jsonl')]
ibm_emnlp_dataset = [json.loads(ln) for ln in open('data/ibm_emnlp_positive_test.jsonl')]

# Divide them
arc_c = []
for sentence in arc_dataset:
    arc_c.append(sentence["claim"])

# Divide them
ibm_c = []
for sentence in ibm_emnlp_dataset:
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
import gensim

from gensim.models import Word2Vec
if not os.path.exists('/home/keshavsingh/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
    
model = gensim.models.KeyedVectors.load_word2vec_format('/home/keshavsingh/GoogleNews-vectors-negative300.bin.gz', binary=True)

f = open('matched_claim_ibm_emnlp_test.txt', 'w')
for sentence in ibm_c:
    dist = []
    for sent1 in arc_c:
        distance = model.wmdistance(sentence.lower().split(), sent1.lower().split())
        dist.append(distance)
    index = dist.index(min(dist))
    f.write(arc_c[index] + '\n')
#    print(min(dist), sentence, arc_c[index])
f.close()

