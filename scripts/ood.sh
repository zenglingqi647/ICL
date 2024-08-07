tmux
tmux attach -t 
tmux kill-session

cd /data1/lzengaf/cs182/ICL/src
conda activate in-context-learning
export CUDA_VISIBLE_DEVICES=

python train.py --config conf/ood/standard.yaml
python train.py --config conf/ood/opposite.yaml
python train.py --config conf/ood/random.yaml
python train.py --config conf/ood/orthogonal.yaml
python train.py --config conf/ood/proj.yaml

python train.py --config conf/ood/standard_rbf.yaml
python train.py --config conf/ood/opposite_rbf.yaml
python train.py --config conf/ood/random_rbf.yaml
python train.py --config conf/ood/orthogonal_rbf.yaml
python train.py --config conf/ood/proj_rbf.yaml

# eval
export PYTHONPATH=/data1/lzengaf/cs182/ICL/src
python /data1/lzengaf/cs182/ICL/src/analysis/ood.py