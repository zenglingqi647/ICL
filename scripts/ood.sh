tmux

cd /data1/lzengaf/cs182/ICL/src
conda activate in-context-learning
export CUDA_VISIBLE_DEVICES=

python train.py --config conf/ood/None.yaml
python train.py --config conf/ood/normal.yaml
python train.py --config conf/ood/permute.yaml
python train.py --config conf/ood/uniform.yaml