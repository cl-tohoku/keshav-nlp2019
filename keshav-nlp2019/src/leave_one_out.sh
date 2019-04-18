for f in `ls /home/keshavsingh/keshav-nlp2019/loodata/train/*`; do
	f_name=`basename $f`
    	echo "Training on file: $f_name"    
    	CUDA_VISIBLE_DEVICES=1 python train.py --train-data /home/keshavsingh/keshav-nlp2019/loodata/train/${f_name}
	echo "Testing on file: $f_name"    
    	CUDA_VISIBLE_DEVICES=1 python test_mrr.py --test-data /home/keshavsingh/keshav-nlp2019/loodata/test/${f_name}
done
