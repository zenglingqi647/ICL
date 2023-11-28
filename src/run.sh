cd /data1/lzengaf/cs182/ICL/src
conda activate in-context-learning
export CUDA_VISIBLE_DEVICES=

python train.py --config conf/logistic_regression.yaml
python train.py --config conf/rbf_logistic_regression.yaml
