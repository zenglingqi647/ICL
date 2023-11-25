import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import warnings
import xgboost as xgb

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import tree
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from base_models import NeuralNetwork, ParallelNetworks

from tqdm import tqdm
import torch
import yaml
import numpy as np


def from_numpy(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif np.isscalar(x):
        return torch.tensor(x).float()
    else:
        raise ValueError("Unknown type: {}".format(type(x)))


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
        ],
        "logistic_regression": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (LDAModel, {}),
            (SVMModel, {}),
        ],
        "rbf_classification": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (GPModel, {}),
            (RBFGPModel, {
                "length_scale": 1.0
            }),
            (SVMModel, {
                "kernel": 'rbf'
            }),
            (RBFNNModel, {
                "n_neighbors": 3
            }),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
        ] + [(LassoModel, {
            "alpha": alpha
        }) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (DecisionTreeModel, {
                "max_depth": 4
            }),
            (DecisionTreeModel, {
                "max_depth": None
            }),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):

    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


class NNModel:

    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


class RBFNNModel:

    def __init__(self, n_neighbors, gamma=1.0):
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.name = f"RBFNN_n={n_neighbors}_gamma={gamma}"

    def rbf_kernel(self, dist):
        return torch.exp(-self.gamma * dist)

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            # Use RBF kernel for weights
            weights = self.rbf_kernel(dist)
            inf_mask = torch.isinf(weights).float()  # deal with exact match
            inf_row = torch.any(inf_mask, axis=1)
            weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:

    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]

            ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2), driver=self.driver)

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class LogisticModel:

    def __init__(self):
        self.name = f"logistic_regression"
        self.tol = 1e-10

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    if ((train_ys.abs() - 1)**2).mean() > self.tol:
                        pred[j] = torch.zeros_like(train_ys[0])
                    elif len(torch.unique(train_ys)) == 1:
                        # if all labels are the same, just predict that label
                        pred[j] = train_ys[0]
                    else:
                        # unregularized logistic regression
                        clf = LogisticRegression(penalty='none', fit_intercept=False)

                        clf.fit(train_xs, train_ys)

                        # w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                        test_x = xs[j, i:i + 1]
                        y_pred = clf.predict_proba(test_x)
                        # y_pred = (test_x @ w_pred.float()).squeeze(1)
                        # Predict prob normalized to [-1, 1]
                        pred[j] = 2 * y_pred[0, 1] - 1

            preds.append(pred)

        return torch.stack(preds, dim=1)


class RBFLogisticModel:

    def __init__(self, gamma=1.0):
        self.name = "rbf_logistic_regression"
        self.gamma = gamma
        self.tol = 1e-10

    def _apply_rbf_kernel(self, X1, X2):
        # apply the RBF kernel to the data
        return torch.tensor(rbf_kernel(X1, X2, gamma=self.gamma))

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]
                    test_x = xs[j, i:i + 1]

                    if ((train_ys.abs() - 1)**2).mean() > self.tol:
                        pred[j] = torch.zeros_like(train_ys[0])
                    elif len(torch.unique(train_ys)) == 1:
                        # if all labels are the same, just predict that label
                        pred[j] = train_ys[0]
                    else:
                        kernel_train = self._apply_rbf_kernel(train_xs, train_xs)
                        kernel_test = self._apply_rbf_kernel(test_x, train_xs)

                        # train logistic regression on the kernel-transformed data
                        clf = LogisticRegression(penalty='none', fit_intercept=False)
                        clf.fit(kernel_train, train_ys)

                        # predict using the kernel-transformed test data
                        y_pred = clf.predict_proba(kernel_test)
                        pred[j] = 2 * y_pred[0, 1] - 1

            preds.append(pred)

        return torch.stack(preds, dim=1)


class AveragingModel:

    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:

    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter)

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i:i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:

    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(ys.shape[0], self.model_class, **self.model_class_args)
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i:i + 1], ys[:, i:i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[:self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(outputs[:, :, 0], train_ys_cur).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(outputs_test[:, :, 0], test_ys).detach()
                            print(f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}")

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i:i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:

    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i:i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)


class LDAModel:

    def __init__(self):
        self.name = "lda"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # number of class should be >= 2, number of samples should be >= number of classes
                    if train_xs.shape[0] < 3:
                        continue
                    clf = LinearDiscriminantAnalysis()

                    clf.fit(train_xs, train_ys)
                    test_x = xs[j, i:i + 1]
                    y_pred = clf.predict(test_x)

                    pred[j] = from_numpy(y_pred[0])

            preds.append(pred)

        return torch.stack(preds, dim=1)


class SVMModel:

    def __init__(self, kernel='linear', C=1.0):
        # for the rbf kernelized version, use kernel='rbf'
        self.name = f"svm-{kernel}"
        self.kernel = kernel
        self.C = C

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # The number of classes has to be greater than one
                    if len(np.unique(train_ys)) < 2:
                        continue

                    clf = SVC(kernel=self.kernel, C=self.C, probability=True)
                    clf.fit(train_xs, train_ys)
                    test_x = xs[j, i:i + 1]
                    y_pred = clf.predict(test_x)

                    pred[j] = from_numpy(y_pred[0])

            preds.append(pred)

        return torch.stack(preds, dim=1)


class GPModel:

    def __init__(self):
        self.name = "gaussian_process_classifier"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]
                    if len(np.unique(train_ys)) < 2:
                        continue
                    test_x = xs[j, i:i + 1]

                    clf = GaussianProcessClassifier()
                    clf.fit(train_xs, train_ys)

                    y_pred = clf.predict_proba(test_x)
                    pred[j] = y_pred[0, 1]  # Assuming binary classification

            preds.append(pred)

        return torch.stack(preds, dim=1)


class RBFGPModel:

    def __init__(self, kernel=None, length_scale=1.0):
        self.name = "gaussian_process_classifier"
        self.kernel = kernel if kernel is not None else 1.0 * RBF(length_scale=length_scale)

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]
                    if len(np.unique(train_ys)) < 2:
                        continue
                    test_x = xs[j, i:i + 1]

                    clf = GaussianProcessClassifier(kernel=self.kernel)
                    clf.fit(train_xs, train_ys)

                    y_pred = clf.predict_proba(test_x)
                    pred[j] = y_pred[0, 1]  # Assuming binary classification

            preds.append(pred)

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    from tasks import RBFClassification

    task = RBFClassification(5, 64)
    xs = torch.normal(torch.zeros((64, 11, 5)))
    ys = task.evaluate(xs)

    logistic_model_bl = [NNModel(n_neighbors=3), LDAModel(), SVMModel()]
    rbf_logistic_model_bl = [
        NNModel(n_neighbors=3),
        RBFNNModel(n_neighbors=3),
        SVMModel(kernel='rbf'),
        GPModel(),
        RBFGPModel(length_scale=1.0)
    ]

    print('test vanilla logistic regression')
    for (model, name) in zip(logistic_model_bl, ['NNModel(n_neighbors=3)', 'LDAModel()', 'SVMModel()']):
        # if name == 'LDAModel()':
        #     continue
        pred = model(xs, ys).detach()
        metrics = task.get_metric()(pred, ys)
        print(name, torch.histc(metrics, 2))

    print('test rbf logistic regression')
    for (model, name) in zip(rbf_logistic_model_bl, [
            'NNModel(n_neighbors=3)', 'RBFNNModel(n_neighbors=3)', 'SVMModel(kernel=rbf)', 'GPModel()',
            'RBFGPModel(length_scale=1.0)'
    ]):
        pred = model(xs, ys).detach()
        metrics = task.get_metric()(pred, ys)
        print(name, torch.histc(metrics, 2))
