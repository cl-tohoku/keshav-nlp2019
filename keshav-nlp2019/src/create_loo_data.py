import sys
import json
import argparse

def main(args):
    print "loading data"
    training_dataset = [json.loads(ln) for ln in open(args.train_data)]
    testing_dataset = [json.loads(ln) for ln in open(args.test_data)]	
    topics = set()
    
    for row in training_dataset:
        topics.add(row['topic'])

    for i, topic in enumerate(topics):
        test_output = []
        train_output = []
        for row in testing_dataset:
            if row["topic"] == topic:
                test_output.append(row)
        for row in training_dataset:
            if row["topic"] != topic:
                train_output.append(row)
        with open('/home/keshavsingh/keshav-nlp2019/loodata/train/%s.jsonl'%(str(i)), 'w') as train:
            for dic in train_output:
            	json.dump(dic, train)
                train.write("\n")
	with open('/home/keshavsingh/keshav-nlp2019/loodata/test/%s.jsonl'%(str(i)), 'w') as test:
            for dic in test_output:
            	json.dump(dic, test)
                test.write("\n")

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data','--train-data', dest='train_data', required=True,
        help="Training data.")
    parser.add_argument(
        '-test','--test-data',dest='test_data',required=True,
        help="Testing data.")
    args = parser.parse_args()
    main(args)
