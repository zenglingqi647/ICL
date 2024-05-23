"""
Functions for generating different kinds of train - test data
"""
from typing import Tuple
import torch
import matplotlib.pyplot as plt
from samplers import get_data_sampler, sample_transformation
from tqdm import tqdm
# TODO: args passing


def project_onto_hyperplane(data: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """
    Projects data onto the hyperplane defined by the given normal vector.

    Args:
        data (torch.Tensor): Data to be projected. Only the first `n_dims_truncated` dimensions are considered for projection.
        normal (torch.Tensor): Normal vector of the hyperplane. The dimension of the normal vector is the same as `n_dims_truncated`.

    Returns:
        torch.Tensor: Projected data.
    """
    subsp_data = data[..., :normal.shape[0]]
    return_data = torch.zeros_like(data)

    normal = normal / torch.norm(normal)  # Ensure the normal is a unit vector
    projection_matrix = torch.eye(subsp_data.shape[-1]) - torch.outer(normal, normal)
    data_projected = subsp_data @ projection_matrix.t()
    return_data[..., :normal.shape[0]] = data_projected
    return return_data


def gen_standard(data_sampler,
                 n_points: int,
                 b_size: int,
                 n_dims_truncated: int = None,
                 seeds: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate standard train-test data.

    Args:
        data_sampler (DataSampler): DataSampler object.
        n_points (int): Number of points to sample.
        b_size (int): Batch size.
        n_dims_truncated (int): Number of dimensions to sample.. Defaults to None.
        seeds (int): Random seed. Defaults to None.

    Returns:
        Tuple: Tuple of train and test data (torch.Tensor, torch.Tensor)
    """
    xs_train = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_test = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    return xs_train, xs_test


def gen_opposite_orthant(data_sampler,
                         n_points: int,
                         b_size: int,
                         n_dims_truncated: int = None,
                         seeds: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train-test data that randomly distributed in opposite orthant.

    Args:
        data_sampler (DataSampler): DataSampler object.
        n_points (int): Number of points to sample.
        b_size (int): Batch size.
        n_dims_truncated (int): Number of dimensions to sample.. Defaults to None.
        seeds (int): Random seed. Defaults to None.

    Returns:
        Tuple: Tuple of train and test data (torch.Tensor, torch.Tensor)
    """
    xs_train = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_test = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)

    # TODO: n_dims should be passed in
    pattern = torch.sign(torch.randn(20))
    assert xs_train.shape[-1] == pattern.shape[0] and xs_test.shape[-1] == pattern.shape[
        0], "Pattern dimension must match data dimension."
    pattern = pattern.view(1, 1, -1)
    xs_train = xs_train.abs() * pattern
    xs_test = xs_test.abs() * -pattern
    print(pattern)
    return xs_train, xs_test


def gen_random_orthant(data_sampler,
                       n_points: int,
                       b_size: int,
                       n_dims_truncated: int = None,
                       seeds: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate train-test data that randomly distributed in orthant. The process is the same for each sample in the batch.

    Args:
        data_sampler (DataSampler): DataSampler object.
        n_points (int): Number of points to sample.
        b_size (int): Batch size.
        n_dims_truncated (int): Number of dimensions to sample.. Defaults to None.
        seeds (int): Random seed. Defaults to None.

    Returns:
        Tuple: Tuple of train and test data (torch.Tensor, torch.Tensor)
    """
    xs_train = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_test = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)

    max_perm = 2 * n_points
    perm_set = set()
    dim_signs = torch.sign(torch.randn(n_dims_truncated))
    for _ in range(max_perm):
        init_tensor = torch.zeros(20)
        init_tensor[:n_dims_truncated] = dim_signs[torch.randperm(dim_signs.shape[0])]
        perm_set.add(init_tensor)
        if len(perm_set) >= max_perm:
            break

    permutations = torch.stack(list(perm_set)).type(torch.float32)

    xs_train = xs_train.abs() * permutations[:n_points]
    xs_test = xs_test.abs() * permutations[n_points:]
    return xs_train, xs_test


def gen_orthogonal(data_sampler,
                   n_points: int,
                   b_size: int,
                   n_dims_truncated: int = None,
                   seeds: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train-test data that are orthogonally distributed in the subspace spanned by the first n_dims_truncated.

    Args:
        data_sampler (DataSampler): DataSampler object.
        n_points (int): Number of points to sample.
        b_size (int): Batch size.
        n_dims_truncated (int): Number of dimensions to sample. Defaults to None.
        seeds (int): Random seed. Defaults to None.

    Returns:
        Tuple: Tuple of train and test data (torch.Tensor, torch.Tensor)
    """
    if seeds is not None:
        torch.manual_seed(seeds)
    if n_dims_truncated is None:
        n_dims_truncated = xs_train.shape[-1]
    # TODO: n_dims should be passed in
    assert n_dims_truncated <= 20, "n_dims_truncated should be less than or equal to 20"

    xs_train = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_test = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)

    normal_vector = torch.randn(n_dims_truncated)
    normal_vector = normal_vector / torch.norm(normal_vector)  # Normalize to unit vector

    orthogonal_normal = torch.randn(n_dims_truncated)
    orthogonal_normal -= orthogonal_normal @ normal_vector * normal_vector  # Make orthogonal to normal_vector
    orthogonal_normal = orthogonal_normal / torch.norm(orthogonal_normal)  # Normalize to unit vector

    xs_train_projected = project_onto_hyperplane(xs_train, normal_vector)
    xs_test_projected = project_onto_hyperplane(xs_test, orthogonal_normal)

    return xs_train_projected, xs_test_projected


def gen_projection(data_sampler,
                   n_points: int,
                   b_size: int,
                   n_dims_truncated: int = None,
                   seeds: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train-test data pair. The training data is projected onto a random hyperplane, which has dimensions of n_dims_truncated // 2 + 1, and the test data is distributed in the entire space.

    Args:
        data_sampler (DataSampler): DataSampler object.
        n_points (int): Number of points to sample.
        b_size (int): Batch size.
        n_dims_truncated (int): Number of dimensions to sample. Defaults to None.
        seeds (int): Random seed. Defaults to None.

    Returns:
        Tuple: Tuple of train and test data (torch.Tensor, torch.Tensor)
    """
    if seeds is not None:
        torch.manual_seed(seeds)
    if n_dims_truncated is None:
        n_dims_truncated = xs_train.shape[-1]
    # TODO: n_dims should be passed in
    assert n_dims_truncated <= 20, "n_dims_truncated should be less than or equal to 20"

    xs_train = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)
    xs_test = data_sampler.sample_xs(n_points, b_size, n_dims_truncated, seeds)

    n_dims_to_zero = n_dims_truncated - (n_dims_truncated//2 + 1)
    zero_indices = torch.randperm(n_dims_truncated)[:n_dims_to_zero]

    mask = torch.ones_like(xs_train)
    mask[..., n_dims_truncated:] = 0
    mask[..., zero_indices] = 0

    return xs_train * mask, xs_test


def gen_expansion(data_sampler,
                  n_points: int,
                  b_size: int,
                  n_dims_truncated: int = None,
                  seeds: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    task = "projection"
    func_dict = {
        # "standard": gen_standard,
        "opposite_orthant": gen_opposite_orthant,
        "random_orthant": gen_random_orthant,
        "orthogonal": gen_orthogonal,
        "projection": gen_projection,
        "expansion": gen_expansion
    }
    for i in tqdm(range(1)):
        xs, test_xs = func_dict[task](data_sampler, 40, 1, 6)
        # xs, test_xs = func_dict[task](data_sampler, 40, 1, 20)
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
