"""
Weights & Biases sweep over the ridge regularization strength (lambda),
replacing train.py's manual logspace grid search with a Bayesian search.

Setup:
    pip install wandb
    wandb login

Run:
    wandb sweep sweep.yaml          # prints a sweep ID
    wandb agent <sweep_id>          # launches runs against it

Each agent run logs cv_mae_kcal_mol (the sweep's optimization target),
test_mae_kcal_mol, and the sampled lambda to your W&B project dashboard.
"""
import numpy as np
import wandb
from scipy import io
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from train import RidgeRegression, build_features, cross_validate

DEFAULT_CONFIG = {
    "data": "../data/qm7.mat",
    "log_lambda": -5.0,
    "n_folds": 5,
    "test_size": 0.2,
    "seed": 42,
}


def main():
    run = wandb.init(config=DEFAULT_CONFIG)
    cfg = run.config

    np.random.seed(cfg["seed"])
    data = io.loadmat(cfg["data"])
    atomic_nums = np.array(data["Z"])
    atomization_energies = np.array(data["T"]).reshape(-1, 1)

    X = build_features(atomic_nums)
    X_centered = X - X.mean(axis=0, keepdims=True)
    T_centered = atomization_energies - atomization_energies.mean()

    lambd = 10 ** cfg["log_lambda"]
    cv_mae, cv_std = cross_validate(
        X_centered, T_centered, lambd, RidgeRegression(), n_folds=cfg["n_folds"]
    )

    X_train, X_test, t_train, t_test = train_test_split(
        X_centered, T_centered, test_size=cfg["test_size"], random_state=cfg["seed"]
    )
    model = RidgeRegression()
    model.fit(X_train, t_train, lambd)
    test_mae = mean_absolute_error(t_test, model.predict(X_test))

    wandb.log({
        "lambda": lambd,
        "cv_mae_kcal_mol": cv_mae,
        "cv_std_kcal_mol": cv_std,
        "test_mae_kcal_mol": float(test_mae),
    })


if __name__ == "__main__":
    main()
