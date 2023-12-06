tmux

cd /data1/lzengaf/cs182/ICL/src
conda activate in-context-learning
export CUDA_VISIBLE_DEVICES=

python train.py --config conf/randlb/None.yaml
python train.py --config conf/randlb/normal.yaml
python train.py --config conf/randlb/permute.yaml
python train.py --config conf/randlb/uniform.yaml

python train.py --config conf/randlb/None_rbf.yaml
python train.py --config conf/randlb/normal_rbf.yaml
python train.py --config conf/randlb/permute_rbf.yaml
python train.py --config conf/randlb/uniform_rbf.yaml

export PYTHONPATH=/data1/lzengaf/cs182/ICL/src
python /data1/lzengaf/cs182/ICL/src/analysis/randlb.py