import json
import sys
# Load benchmark.
#arc_dataset = [json.loads(ln) for ln in open('data/complete_arc.jsonl')]
#ibm_acl_dataset = [json.loads(ln) for ln in open('data/ibm_acl_positive.jsonl')]

arc_c = []
ibm_c = []


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


from collections import defaultdict

distances = defaultdict(list)

f = open("data/sentences/output/"+sys.argv[1], 'w')
for line in sys.stdin:
    fields = line.split("\t")
    ibm_c = fields[0]
    arc_c = fields[1]
    distances[ibm_c].append(model.wmdistance(ibm_c.lower().split(), arc_c.lower().split()))

for k,v in distances.items():
    index = distances[k].index(min(distances[k]))
    f.write(k + "\t" + str(distances[k][index]) + "\n")
f.close()

