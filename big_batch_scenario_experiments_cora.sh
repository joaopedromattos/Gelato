# Big Batch Scenario - Batched Gelato
# Scalling parameter 2 - Batch-size 0.1
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 1 --scaling-parameter 2 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 2 --scaling-parameter 2 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 3 --scaling-parameter 2 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 4 --scaling-parameter 2 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 5 --scaling-parameter 2 --epochs 150 --cuda 0

# Scalling parameter 3 - Batch-size 0.1
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 1 --scaling-parameter 3 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 2 --scaling-parameter 3 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 3 --scaling-parameter 3 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 4 --scaling-parameter 3 --epochs 150 --cuda 0
python3 src/train.py --train-batch-ratio 0.1 --dataset Cora --batch-version true --random-seed 5 --scaling-parameter 3 --epochs 150 --cuda 0
