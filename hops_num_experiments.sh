# Batch Gelato
# Scalling parameter 3 - Batch-size 0.001
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 1 --scaling-parameter 3 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 2 --scaling-parameter 3 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 3 --scaling-parameter 3 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 4 --scaling-parameter 3 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 5 --scaling-parameter 3 --epochs 15 --cuda 1

# Scalling parameter 2 - Batch-size 0.001
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 1 --scaling-parameter 2 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 2 --scaling-parameter 2 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 3 --scaling-parameter 2 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 4 --scaling-parameter 2 --epochs 15 --cuda 1
python3 src/train.py --train-batch-ratio 0.001 --dataset Photo --max-neighborhood-size 250 --batch-version true --random-seed 5 --scaling-parameter 2 --epochs 15 --cuda 1
