import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

root = '/data1/lzengaf/cs182/ICL/models/logistic_regression_ood/'


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_train_test_dist(config):
    return config.get('training', {}).get('train_test_dist')

if not os.path.exists(os.path.join('/data1/lzengaf/cs182/ICL', 'plots')):
    os.mkdir(os.path.join('/data1/lzengaf/cs182/ICL', 'plots'))

for run_id in os.listdir(root):
    # Load data from npy files
    config = load_config(os.path.join(root, run_id, 'config.yaml'))
    if not os.path.exists(os.path.join(root, run_id, 'train_xs.npy')):
        print(os.path.join(root, run_id, 'train_xs.npy'))
        continue
    train_data = np.load(os.path.join(root, run_id, 'train_xs.npy'))
    test_data = np.load(os.path.join(root, run_id, 'test_xs.npy'))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.scatter(train_data[:, 0], train_data[:, 1], alpha=0.5)
    plt.title('Scatter Plot of First Dataset')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Create scatter plot for the second dataset
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.5, color='red')
    plt.title('Scatter Plot of Second Dataset')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plots
    plt.tight_layout()
    plt.savefig(f'/data1/lzengaf/cs182/ICL/plots/{get_train_test_dist(config)}.png')
