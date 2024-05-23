from typing import Tuple
import torch
import matplotlib.pyplot as plt
from samplers import get_data_sampler, sample_transformation
from tqdm import tqdm

# Functions for generating different kinds of train - test data


def gen_standard(data_sampler, n_points, b_size, n_dims_truncated=None, seeds=None):
    xs = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    return xs, None


def gen_opposite_quadrants(data_sampler,
                           n_points: int,
                           b_size: int,
                           n_dims_truncated: int = None,
                           seeds: int = None) -> Tuple:
    """
    Generate train-test data that randomly distributed in opposite quadrants.

    Args:
        data_sampler (DataSampler): DataSampler object.
        n_points (int): Number of points to sample.
        b_size (int): Batch size.
        n_dims_truncated (int): Number of dimensions to sample.
        seeds (int): Random seed.

    """
    xs_train = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_test = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    pattern = torch.sign(torch.randn(n_dims_truncated))
    assert xs_train.shape[-1] == pattern.shape[0] and xs_test.shape[-1] == pattern.shape[
        0], "Pattern dimension must match data dimension."
    pattern = pattern.view(1, 1, -1)
    xs_train = xs_train.abs() * pattern
    xs_test = xs_test.abs() * -pattern
    print(pattern)
    return xs_train, xs_test


def gen_random_quadrants(data_sampler, n_points, b_size, n_dims_truncated=None, seeds=None):
    xs = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train = xs.abs() * pattern
    xs_test = xs
    return xs_train, xs_test


def gen_orthogonal(data_sampler, n_points, b_size, n_dims_truncated=None, seeds=None):
    xs = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    xs_train = xs
    xs_test = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_i = xs[:, i:i + 1, :]
        xs_train_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_i, full_matrices=False)
        xs_train_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_i_orthogonalized = (xs_test_i - xs_test_i @ xs_train_i_projection)
        xs_test_i_normalized = (xs_test_i_orthogonalized * xs_test_i.norm(dim=2).unsqueeze(2) /
                                xs_test_i_orthogonalized.norm(dim=2).unsqueeze(2))

        xs_test[:, i:i + 1, :] = xs_test_i_normalized
    return xs_train, xs_test


def gen_projection(data_sampler, n_points, b_size, n_dims_truncated=None, seeds=None):
    xs = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_train = xs
    xs_test = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0).float().unsqueeze(dim=1)
        xs_test[:, i:i + 1, :] = ind_mat @ xs_train_i
    return xs_train, xs_test


def gen_expansion(data_sampler, n_points, b_size, n_dims_truncated=None, seeds=None):
    xs = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_train = xs
    xs_test = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0).float().unsqueeze(dim=1)
        xs_test[:, i:i + 1, :] = ind_mat @ xs_train_i
    return xs_train, xs_test


if __name__ == "__main__":
    data_sampler = get_data_sampler('gaussian', n_dims=20)
    task = "opposite_quadrants"
    func_dict = {
        # "standard": gen_standard,
        "opposite_quadrants": gen_opposite_quadrants,
        "random_quadrants": gen_random_quadrants,
        "orthogonal": gen_orthogonal,
        "projection": gen_projection,
        "expansion": gen_expansion
    }
    for i in tqdm(range(20)):
        xs, test_xs = func_dict[task](data_sampler, 40, 1, 20)
        max_range = max(abs(xs[..., 0].min()), abs(xs[..., 0].max()), abs(xs[..., 1].min()), abs(xs[..., 1].max()),
                        abs(test_xs[..., 0].min()), abs(test_xs[..., 0].max()), abs(test_xs[..., 1].min()),
                        abs(test_xs[..., 1].max()))
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(xs[..., 0], xs[..., 1], alpha=0.5, color='blue', label='Train Data')
        plt.scatter(test_xs[..., 0], test_xs[..., 1], alpha=0.5, color='red', label='Test Data')

        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title('Scatter Plot of Datasets')
        plt.xlabel('dim 0')
        plt.ylabel('dim 1')
        plt.xlim(max_range * -1 - 0.1, max_range + 0.1)
        plt.ylim(max_range * -1 - 0.1, max_range + 0.1)

        plt.tight_layout()
        plt.savefig(f'figs/opposite/{task}_{i}.png')
        plt.close()
