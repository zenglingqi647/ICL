import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from tasks import get_task_sampler


def ood_eval(model,
                conf,run_path,
                n_points=range(10, 80, 10),
                batch_size=64):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # for n_dims in tqdm(n_dims_ls):
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    fig, ax = plt.subplots()
    
    plt.title("Robustness to Query Scales")
    plt.xlabel("scale")
    plt.ylabel("accuracy")
    ax.set_xscale('log')

    for num_ex in tqdm(n_points):
        results = []
        
        task_sampler = get_task_sampler(conf.training.task, n_dims, batch_size, **conf.training.task_kwargs)
        task = task_sampler()
        
        xs = np.load(os.path.join(run_path, 'test_xs.npy'))
        ys = task.evaluate(xs)
        
        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device))
            pred = pred.cpu()
        metric = task.get_metric()
        acc = metric(pred, ys).numpy()
        results.append(acc.mean())
        
        plt.plot(n_points, results, marker='o', linewidth=2, label=f"{num_ex} examples")
    
    plt.legend()
    plt.savefig("figs/ood_accuracy.png", bbox_inches='tight')

    # Plotting
    # for n_dims, losses in results.items():
    #     for i, loss in enumerate(losses):
    #         plt.plot(loss, lw=2, label=f"n_dims={n_dims}, scale={scales[i]}")
    


if __name__ == "__main__":
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')

    run_dir = "/data1/lzengaf/cs182/ICL/models"

    df = read_run_dir(run_dir)
    task = "logistic_regression"
    run_id = "trained_partial"  # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    recompute_metrics = False

    if recompute_metrics:
        get_run_metrics(run_path)  # these are normally precomputed at the end of training
    model, conf = get_model_from_run(run_path)

    ood_eval(model, conf, run_path)