tmux

cd /data1/lzengaf/cs182/ICL/src
conda activate in-context-learning
export CUDA_VISIBLE_DEVICES=

python train.py --config conf/rand_lb/standard.yaml
python train.py --config conf/rand_lb/opposite.yaml
python train.py --config conf/rand_lb/random.yaml
python train.py --config conf/rand_lb/orthogonal.yaml
python train.py --config conf/rand_lb/overlapping.yaml