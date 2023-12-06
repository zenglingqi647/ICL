import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from tasks import get_task_sampler
from samplers import get_data_sampler

def ood_eval(model, conf, run_path, n_points=range(10, 90, 10), batch_size=64):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # for n_dims in tqdm(n_dims_ls):
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    plt.title(conf.training.train_test_dist)
    plt.xlabel("n_points")
    plt.ylabel("accuracy")

    results = []
    for num_ex in tqdm(n_points):

        task_sampler = get_task_sampler(conf.training.task, n_dims, batch_size, **conf.training.task_kwargs)
        task = task_sampler()
        
        xs_path = os.path.join(run_path, 'test_xs.npy')
        if not os.path.exists(xs_path):
            if 'standard' not in conf.training.task:
                raise ValueError('Missing test_xs.npy for OOD task!')
            data_sampler = get_data_sampler(conf.training.data, n_dims)
            xs = data_sampler.sample_xs(b_size=batch_size, n_points=num_ex)
            
        xs = np.load(os.path.join(run_path, 'test_xs.npy'))
        xs = torch.from_numpy(xs[:, :num_ex, :])
        ys = task.evaluate(xs)

        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device))
            pred = pred.cpu()
        metric = task.get_metric()
        acc = metric(pred, ys).numpy()
        results.append(acc.mean())

    plt.plot(list(n_points), results, marker='o', linewidth=2, label="Accuracy")
    plt.legend()
    plt.savefig(f"figs/{conf.training.task}_{conf.training.train_test_dist}.png", bbox_inches='tight')


if __name__ == "__main__":
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')

    run_dir = "/data1/lzengaf/cs182/ICL/models/"
    task = 'logistic_regression_ood'

    assert 'ood' in task, 'Please specify an OOD task!'

    df = read_run_dir(os.path.join(run_dir, task))

    for run_id in tqdm(os.path.join(run_dir, task)):
        if not os.path.isdir(os.path.join(run_dir, task, run_id)):
            continue
        
        run_path = os.path.join(run_dir, task, run_id)
        recompute_metrics = False

        if recompute_metrics:
            get_run_metrics(run_path)  # these are normally precomputed at the end of training
        model, conf = get_model_from_run(run_path)

        ood_eval(model, conf, run_path)