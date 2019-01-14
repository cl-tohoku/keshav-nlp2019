```
python src/generate_data.py > data/benchmark.jsonl
CUDA_VISIBLE_DEVICES=2 python src/train.py --train-data data/benchmark.jsonl
```
