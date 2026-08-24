"""
Train a Ridge regression model on the QM7 dataset to predict molecular
atomization energy from atom-type composition.

This is the Section 3.1 (atom-based) model from the original notebook,
refactored so it can run outside Jupyter, log to MLflow, and save an
artifact that app/serve.py can load.

Usage:
    python train.py --data data/qm7.mat --out model/model.joblib
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from scipy import io
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from model import QM7RidgeModel

# Atom index -> one-hot column. Matches the notebook's mapping.
ATOM_ONE_HOT = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}  # H, C, N, O, S
ATOM_SYMBOLS = ["H", "C", "N", "O", "S"]
REGISTERED_MODEL_NAME = "qm7-atomization-energy"
CHAMPION_ALIAS = "champion"


class RidgeRegression:
    """Closed-form ridge regression, identical to the notebook's implementation."""

    def fit(self, X, t, lambd):
        n, d = X.shape
        c_xx = (1 / n) * np.dot(X.T, X)
        c_xx_lambda = c_xx + (lambd * np.identity(d))
        c_xt = (1 / n) * np.dot(X.T, t)
        self.weights = np.dot(np.linalg.inv(c_xx_lambda), c_xt)
        return self

    def predict(self, X):
        return np.dot(X, self.weights)

    def get_weights(self):
        return self.weights

    @staticmethod
    def get_pred_error(t, preds):
        return mean_absolute_error(t, preds)


def build_features(atomic_nums: np.ndarray) -> np.ndarray:
    """Sum one-hot atom encodings per molecule -> (N, 5) composition vector."""
    n_molecules, max_atoms = atomic_nums.shape
    one_hot = np.zeros((n_molecules, max_atoms, len(ATOM_ONE_HOT)))
    for i, molecule in enumerate(atomic_nums):
        for j, atomic_num in enumerate(molecule):
            if atomic_num in ATOM_ONE_HOT:
                one_hot[i, j, ATOM_ONE_HOT[atomic_num]] = 1
    return one_hot.sum(axis=1)  # (N, 5)


def get_selection_matrix(x, n_folds=10):
    n = x.shape[0] // n_folds
    sel = np.zeros((x.shape[0], n_folds))
    for i in range(n_folds):
        sel[i * n:(i + 1) * n, i] = 1
    return sel


def cross_validate(x, t, lambd, model, n_folds=10):
    sel = get_selection_matrix(x, n_folds)
    errors = []
    for k in range(n_folds):
        x_tr, t_tr = x[sel[:, k] == 0], t[sel[:, k] == 0]
        x_te, t_te = x[sel[:, k] == 1], t[sel[:, k] == 1]
        model.fit(x_tr, t_tr, lambd)
        errors.append(model.get_pred_error(t_te, model.predict(x_te)))
    return float(np.mean(errors)), float(np.std(errors))


def select_optimal_lambda(x, t, lambdas, n_folds=10):
    errors = [cross_validate(x, t, lam, RidgeRegression(), n_folds)[0] for lam in lambdas]
    idx = int(np.argmin(errors))
    return lambdas[idx], errors[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/qm7.mat")
    parser.add_argument("--out", default="model/model.joblib")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlflow-uri", default="sqlite:///mlflow.db",
        help="MLflow tracking URI. Defaults to a local SQLite DB (required for "
             "the model registry — the plain './mlruns' file store is deprecated).",
    )
    parser.add_argument(
        "--skip-register", action="store_true",
        help="Log the run to MLflow but skip registering/promoting a model version.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("qm7-atomization-energy")

    with mlflow.start_run():
        t0 = time.time()
        np.random.seed(args.seed)

        data = io.loadmat(args.data)
        atomic_nums = np.array(data["Z"])
        atomization_energies = np.array(data["T"]).reshape(-1, 1)

        X = build_features(atomic_nums)
        x_mean = X.mean(axis=0, keepdims=True)
        t_mean = atomization_energies.mean()
        X_centered = X - x_mean
        T_centered = atomization_energies - t_mean

        mlflow.log_params({
            "n_folds": args.n_folds,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_molecules": X.shape[0],
            "n_features": X.shape[1],
        })

        lambdas = np.logspace(-20, 10, 31)
        optimal_lambda, cv_mae = select_optimal_lambda(
            X_centered, T_centered, lambdas, n_folds=args.n_folds
        )

        X_train, X_test, t_train, t_test = train_test_split(
            X_centered, T_centered, test_size=args.test_size, random_state=args.seed
        )

        model = RidgeRegression()
        model.fit(X_train, t_train, optimal_lambda)
        preds = model.predict(X_test)
        test_mae = mean_absolute_error(t_test, preds)

        train_seconds = time.time() - t0

        mlflow.log_metric("optimal_lambda", float(optimal_lambda))
        mlflow.log_metric("cv_mae_kcal_mol", cv_mae)
        mlflow.log_metric("test_mae_kcal_mol", float(test_mae))
        mlflow.log_metric("train_seconds", train_seconds)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "weights": model.get_weights(),
            "x_mean": x_mean,
            "t_mean": t_mean,
            "atom_symbols": ATOM_SYMBOLS,
            "optimal_lambda": float(optimal_lambda),
            "test_mae_kcal_mol": float(test_mae),
        }
        joblib.dump(artifact, out_path)
        mlflow.log_artifact(str(out_path))

        metrics_path = out_path.parent / "metrics.json"
        metrics_path.write_text(json.dumps(
            {"optimal_lambda": float(optimal_lambda),
             "cv_mae_kcal_mol": cv_mae,
             "test_mae_kcal_mol": float(test_mae),
             "train_seconds": train_seconds},
            indent=2,
        ))

        print(f"Optimal lambda: {optimal_lambda:.4g}")
        print(f"CV MAE: {cv_mae:.2f} kcal/mol")
        print(f"Test MAE: {test_mae:.2f} kcal/mol")
        print(f"Model saved to: {out_path}")

        promoted = False
        if not args.skip_register:
            promoted = register_and_maybe_promote(
                out_path=out_path, test_mae=float(test_mae), run_id=mlflow.active_run().info.run_id
            )

        # Emit a GitHub Actions output so the workflow can decide whether to
        # build and push a new image, without scraping stdout.
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as fh:
                fh.write(f"promoted={'true' if promoted else 'false'}\n")
                fh.write(f"test_mae={test_mae:.4f}\n")


def register_and_maybe_promote(out_path: Path, test_mae: float, run_id: str) -> bool:
    """Register this run's model as a new version, and promote it to the
    `champion` alias only if it doesn't regress the current champion's MAE.

    Returns True if this version was promoted.
    """
    client = MlflowClient()

    # Log the model itself (not just the raw joblib artifact) so it's
    # loadable via the standard mlflow.pyfunc interface, e.g.
    # mlflow.pyfunc.load_model("models:/qm7-atomization-energy/champion").
    sample_input = pd.DataFrame([{"H": 4, "C": 1, "N": 0, "O": 0, "S": 0}])
    sample_output = np.array([-400.0])
    signature = infer_signature(sample_input, sample_output)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=QM7RidgeModel(),
        artifacts={"model": str(out_path)},
        signature=signature,
        input_example=sample_input,
    )
    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        pass  # already exists

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME, source=model_uri, run_id=run_id
    )

    promote = True
    reason = "no existing champion"
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        champion_run = client.get_run(champion.run_id)
        champion_mae = champion_run.data.metrics.get("test_mae_kcal_mol")
        if champion_mae is not None and test_mae > champion_mae:
            promote = False
            reason = f"MAE {test_mae:.2f} did not beat champion's {champion_mae:.2f}"
        else:
            reason = f"MAE {test_mae:.2f} beats or matches champion's {champion_mae:.2f}"
    except MlflowException:
        pass  # no champion registered yet

    if promote:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS, model_version.version
        )
        # Keep the artifact the Dockerfile/serve.py actually read in sync
        # with whichever version is champion.
        champion_path = out_path.parent / "champion.joblib"
        shutil.copy(out_path, champion_path)

    print(f"Registered version {model_version.version}. "
          f"Promoted: {promote} ({reason})")
    return promote


if __name__ == "__main__":
    main()
