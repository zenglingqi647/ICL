
# Theoretical/Empirical Study of In-Context Learning
### **Final project for COMPSCI 182/282A - Designing, Visualizing and Understanding Deep Neural Networks (Fa23)**

## Abstract
This study investigates the in-context learning capabilities of Generative Pre-trained Transformers (GPT) models in performing logistic regression, including its kernelized variant using the Radial Basis Function (RBF) kernel. We explore whether GPT models can effectively learn logistic regression in-context, comparing their performance with traditional machine learning algorithms like k-NN, SVM, and Gaussian Process Classifier. Our methodology involves generating synthetic data for logistic regression tasks and training GPT models on these datasets, both with and without noise. We also examine the impact of scaling on model accuracy. Our findings indicate that GPT models show promising results in learning logistic regression in-context, outperforming or matching most baselines in various scenarios. This research contributes to understanding the potential of GPT models in statistical tasks and opens avenues for further exploration into their in-context learning mechanisms. 

## Setup
The conda environment for this project can be installed using the following command (Linux):
```
conda env create -f environment.yml
```

## Acknowledgements
The code for this project is based on the following repositories:

https://github.com/dtsip/in-context-learning
