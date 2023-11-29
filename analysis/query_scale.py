from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from samplers import get_data_sampler
from tasks import get_task_sampler


def query_scale(model,
                conf,
                scales=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16],
                n_dims_ls=[5, 10, 20, 40, 80],
                n_points=1280,
                batch_size=64):
    results = {n_dims: [] for n_dims in n_dims_ls}

    for n_dims in tqdm(n_dims_ls):
        for scale in scales:
            data_sampler = get_data_sampler(conf.training.data, n_dims)
            task_sampler = get_task_sampler(conf.training.task, n_dims, batch_size, **conf.training.task_kwargs)
            task = task_sampler()
            xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)
            xs = xs * scale
            ys = task.evaluate(xs)
            with torch.no_grad():
                pred = model(xs, ys)

            metric = task.get_metric()
            loss = metric(pred, ys).numpy()

            results[n_dims].append(loss.mean(axis=0))

    # Plotting
    for n_dims, losses in results.items():
        for i, loss in enumerate(losses):
            plt.plot(loss, lw=2, label=f"n_dims={n_dims}, scale={scales[i]}")

    plt.xlabel("# in-context examples")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("figs/accuracy.png", bbox_inches='tight')


if __name__ == "__main__":
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')

    run_dir = "/data1/lzengaf/cs182/ICL/models"

    df = read_run_dir(run_dir)
    task = "rbf_logistic_regression"
    run_id = "trained_partial"  # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    recompute_metrics = False

    if recompute_metrics:
        get_run_metrics(run_path)  # these are normally precomputed at the end of training
    model, conf = get_model_from_run(run_path)

    query_scale(model, conf)