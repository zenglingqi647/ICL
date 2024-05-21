import matplotlib.pyplot as plt
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler
from eval import gen_opposite_quadrants, gen_proj_train_test, gen_random_quadrants, gen_standard, gen_orthogonal_train_test

data_sampler = get_data_sampler('gaussian', n_dims=20)
xs, test_xs = gen_proj_train_test(data_sampler, 40, 1, 20)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(xs[:, 0], xs[:, 1], alpha=0.5)
plt.title('Scatter Plot of First Dataset')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.subplot(1, 2, 2)
plt.scatter(test_xs[:, 0], test_xs[:, 1], alpha=0.5, color='red')
plt.title('Scatter Plot of Second Dataset')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# plt save fig to png
plt.tight_layout()
plt.savefig('proj.png')

