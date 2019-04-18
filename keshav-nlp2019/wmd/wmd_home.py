import json
# Load benchmark.
arc_dataset = [json.loads(ln) for ln in open('data/complete_arc.jsonl')]
new_data = [json.loads(ln) for ln in open('/home/keshavsingh/keshav-nlp2019/new_data/train_data')]

# Divide them
arc_c = []
arc_w = []
for sentence in arc_dataset:
    arc_c.append(sentence["claim"])
    arc_w.append(sentence["warrant"])

# Divide them
ibm_c = []
for sentence in new_data:
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

for claim in ibm_c:
    dist = []
    for clm in arc_c:
        distance = model.wmdistance(sentence.lower().split(), sent1.lower().split())
        dist.append(distance)
    index = dist.index(min(dist))
    f.write(arc_w[index] + '\n')
#    print(min(dist), sentence, arc_c[index])
f.close()

