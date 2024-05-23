
# Theoretical/Empirical Study of In-Context Learning
### **Final project for COMPSCI 182/282A - Designing, Visualizing and Understanding Deep Neural Networks (Fa23)**

## Abstract
This study investigates the in-context learning capabilities of Generative Pre-trained Transformers (GPT) models in performing logistic regression, including its kernelized variant using the Radial Basis Function (RBF) kernel. We explore whether GPT models can effectively learn logistic regression in-context, comparing their performance with traditional machine learning algorithms like k-NN, SVM, and Gaussian Process Classifier. Our methodology involves generating synthetic data for logistic regression tasks and training GPT models on these datasets, both with and without noise. We also examine the impact of scaling on model accuracy. Our findings indicate that GPT models show promising results in learning logistic regression in-context, outperforming or matching most baselines in various scenarios. This research contributes to understanding the potential of GPT models in statistical tasks and opens avenues for further exploration into their in-context learning mechanisms. 

## Setup
The conda environment for this project can be installed using the following command (Linux):
```bash
conda env create -f environment.yml
```

install plotly and kaleido:
```bash
pip install plotly
pip install -U kaleido
pip install nbformat

```
<!-- create from scratch:
```bash
conda create -n icl python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install tqdm wandb matplotlib pandas -c conda-forge -y
# pip install quinine
conda install transformers -c conda-forge -y
pip install pandas
conda install omegaconf -c conda-forge -y
``` -->

## Run the Code
Setup:
```bash
cd /Your_src_folder_path
conda activate icl
export CUDA_VISIBLE_DEVICES=Your_GPU_ID
```

### Basic Experiments
Vanilla Logistic Regression:
```bash
python train.py --config conf/logistic_regression.yaml
```

RBF Logistic Regression with Clean Training and Testing Data:
```bash
python train.py --config conf/rbf_logistic_regression.yaml
```

### Add Noise
Vanilla and RBF Logistic Regression with Noisy Training and Testing Data:
```bash
python train.py --config conf/lr_noise0.2.yaml
python train.py --config conf/rbf_lr_noise0.2.yaml
```
We also experimented noise probabilities of 0.05 and 0.1, which could be run by replacing the 0.2 with 0.05 and 0.1

### Varying problem dimensions
```bash
python train.py --config conf/rbf_lr_noise0.1_dim[10/30/40].yaml
```

### Varying model capacity
```bash
python train.py --config conf/rbf_lr_[small/tiny]_noise0.1.yaml
```

### Scaling
Modify the **task** and **run_id** in /ICL/src/analysis/query_scale.py, and run
```bash
python /ICL/src/analysis/query_scale.py
```

### Data Distribution
Train:
The working directory should be **Your_path/ICL/src**
```bash
# logistic regression
python train.py --config conf/ood/standard.yaml
python train.py --config conf/ood/opposite.yaml
python train.py --config conf/ood/random.yaml
python train.py --config conf/ood/orthogonal.yaml
python train.py --config conf/ood/proj.yaml

# rbf logistic regression
python train.py --config conf/ood/standard_rbf.yaml
python train.py --config conf/ood/opposite_rbf.yaml
python train.py --config conf/ood/random_rbf.yaml
python train.py --config conf/ood/orthogonal_rbf.yaml
python train.py --config conf/ood/proj_rbf.yaml
```
Evaluate and plot:
The working directory should be **Your_path/ICL/**
```bash
export PYTHONPATH={YOUR_WORKING_DIR}/ICL/src
python {YOUR_WORKING_DIR}/ICL/src/analysis/ood.py
```

### Random label
Train:
The working directory should be **Your_path/ICL/src/**
```bash
# logistic regression
python train.py --config conf/randlb/None.yaml
python train.py --config conf/randlb/normal.yaml
python train.py --config conf/randlb/permute.yaml
python train.py --config conf/randlb/uniform.yaml

# rbf logistic regression
python train.py --config conf/randlb/None_rbf.yaml
python train.py --config conf/randlb/normal_rbf.yaml
python train.py --config conf/randlb/permute_rbf.yaml
python train.py --config conf/randlb/uniform_rbf.yaml
```
Evaluate and plot:
The working directory should be **Your_path/ICL/**
```bash
export PYTHONPATH={YOUR_WORKING_DIR}/ICL/src
python {YOUR_WORKING_DIR}/ICL/src/analysis/randlb.py
```

## Testing
ood data generation and visualization:
```bash
export PYTHONPATH=/csproject/t3_lzengaf/lzengaf/ICL/src
python src/ood_data_gen.py
```

## Code Reading Note
Curriculum:
> n_dims start=5, ends=20

Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
> if the label really matter. Change the label to random number. From the training code, we change the target but keep the correct ys to see if the claim is true.
the **point_wise_loss** is not used for gradient update, but for logging. Hence, only the **loss** is modified.

## Acknowledgements
The code for this project is based on the following repositories:

https://github.com/dtsip/in-context-learning

