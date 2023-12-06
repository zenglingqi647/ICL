import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import numpy as np
from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from eval import gen_standard, gen_opposite_quadrants, gen_random_quadrants, gen_overlapping_train_test, gen_orthogonal_train_test

import wandb

torch.backends.cudnn.benchmark = True
DATAGEN_DICT = {
    "standard": gen_standard,
    "opposite": gen_opposite_quadrants,
    "random": gen_random_quadrants,
    "orthogonal": gen_orthogonal_train_test,
    "overlapping": gen_overlapping_train_test
}


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def train_random_label(model, xs, ys, optimizer, loss_func, random_scheme="normal"):
    """
    If random_scheme is "normal", then the random labels are sampled from a normal distribution.
    elif random_scheme is "uniform", then the random labels are sampled from a uniform distribution.
    else, the random labels are permuted from the original labels.
    """
    optimizer.zero_grad()
    output = model(xs, ys)

    if random_scheme == "normal":
        rand_y = torch.randn_like(ys)
    elif random_scheme == "uniform":
        rand_y = torch.rand_like(ys)
    elif random_scheme == "permute":
        rand_y = ys[torch.randperm(ys.shape[0])]
    else:
        pass

    loss = loss_func(output, rand_y)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # train/test distribution
        xs, test_xs = DATAGEN_DICT[args.training.train_test_dist](
            data_sampler,
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        if args.training.random_labels != "None":
            loss, output = train_random_label(model, xs.cuda(), ys.cuda(), optimizer, loss_func, random_scheme=args.training.random_labels)
        else:
            loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (sum(max(curriculum.n_dims_truncated - ii, 0) for ii in range(curriculum.n_points)) /
                         curriculum.n_points)

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(zip(point_wise_tags,
                                               point_wise_loss.cpu().numpy())),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (args.training.keep_every_steps > 0 and i % args.training.keep_every_steps == 0 and not args.test_run and
                i > 0):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

    if test_xs is not None:
        np.save(os.path.join(args.out_dir, 'test_xs.npy'), test_xs.cpu().numpy())
    else:
        np.save(os.path.join(args.out_dir, 'test_xs.npy'), xs.cpu().numpy())
    np.save(os.path.join(args.out_dir, 'train_xs.npy'), xs.cpu().numpy())


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
