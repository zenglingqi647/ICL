import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from tasks import get_task_sampler
from samplers import get_data_sampler

palette = sns.color_palette('colorblind')
PLT_COLOR = [
    palette[0], palette[1], palette[2], palette[3], palette[4], palette[5], palette[6], palette[7], palette[8],
    palette[9]
]


def randlb_eval(model, conf, n_points=range(10, 90, 10), batch_size=64, color=None, title=None):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # for n_dims in tqdm(n_dims_ls):
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    # _ = plt.subplots()
    plt.title(title)
    plt.xlabel("n_points")
    plt.ylabel("accuracy")

    results = []
    for num_ex in tqdm(n_points):

        task_sampler = get_task_sampler(conf.training.task, n_dims, batch_size, **conf.training.task_kwargs)
        task = task_sampler()

        data_sampler = get_data_sampler(conf.training.data, n_dims)
        xs = data_sampler.sample_xs(b_size=batch_size, n_points=num_ex)
        ys = task.evaluate(xs)

        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device))
            pred = pred.cpu()
        metric = task.get_metric()
        acc = metric(pred, ys).numpy()
        results.append(acc.mean())

    plt.plot(list(n_points), results, marker='o', linewidth=2, label=conf.training.random_labels, color=color)
    plt.legend()
    # plt.savefig(f"figs/randlb_{conf.training.task}_{conf.training.random_labels}.png", bbox_inches='tight')


if __name__ == "__main__":
    sns.set_theme('notebook', 'darkgrid')
    run_dir = "/data1/lzengaf/cs182/ICL/models/"
    task = 'rbf_logistic_regression_randlb'

    assert 'randlb' in task, 'Please specify an randlb task!'

    df = read_run_dir(run_dir)

    i = 0
    for run_id in tqdm(os.listdir(os.path.join(run_dir, task)), desc='run'):
        if not os.path.isdir(os.path.join(run_dir, task, run_id)):
            continue

        run_path = os.path.join(run_dir, task, run_id)
        recompute_metrics = False

        if recompute_metrics:
            get_run_metrics(run_path)  # these are normally precomputed at the end of training
        model, conf = get_model_from_run(run_path)

        randlb_eval(model, conf, color=PLT_COLOR[i], title=task)
        i += 1

    plt.savefig(f"figs/randlb_{conf.training.task}.png", bbox_inches='tight')