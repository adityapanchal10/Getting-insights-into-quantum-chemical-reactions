"""
MLflow pyfunc wrapper around the QM7 ridge regression artifact.

Wrapping it this way lets the model be registered, versioned, and loaded
back via `mlflow.pyfunc.load_model("models:/qm7-atomization-energy/champion")`
- the standard interface regardless of what's actually inside (today a
closed-form ridge model, tomorrow maybe something else entirely).
"""
import joblib
import mlflow.pyfunc
import numpy as np

FEATURE_COLUMNS = ["H", "C", "N", "O", "S"]


class QM7RidgeModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.artifact = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        X = model_input[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_centered = X - self.artifact["x_mean"]
        preds = np.dot(X_centered, self.artifact["weights"]) + self.artifact["t_mean"]
        return preds.flatten()
