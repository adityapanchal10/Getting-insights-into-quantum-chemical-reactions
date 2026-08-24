# QM7 atomization energy predictor — Kubernetes + MLOps prototype

Ridge regression model from `Getting Insights into Quantum-Chemical Reactions.ipynb`, refactored into a trainable script, an MLflow-tracked experiment, a FastAPI service, and a Kubernetes deployment with autoscaling.

Verified locally: training reproduces the notebook's ~15.5 kcal/mol test MAE, and the FastAPI service returns correct predictions.

## Project layout

```
qm7-mlops/
├── app/
│   ├── train.py       # training + MLflow tracking + registry/promotion
│   ├── model.py        # pyfunc wrapper for MLflow registry logging
│   ├── serve.py         # FastAPI serving layer (loads champion.joblib)
│   ├── sweep.py           # W&B hyperparameter sweep entrypoint
│   └── sweep.yaml          # W&B sweep search space
├── .github/workflows/
│   └── retrain.yml           # CI: retrain -> gate -> build & push on promotion
├── data/
│   └── qm7.mat                 # QM7 dataset
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── model/                        # created by train.py
├── Dockerfile
├── requirements.txt
├── loadtest.py
└── README.md
```

## 0. Prerequisites (install once)

- Docker Desktop (or Docker Engine)
- `kind` — https://kind.sigs.k8s.io/docs/user/quick-start/#installation
- `kubectl` — https://kubernetes.io/docs/tasks/tools/
- Python 3.11+
- (for the load test) `pip install requests`

## 1. Train the model and track it with MLflow

```bash
cd qm7-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python app/train.py --data data/qm7.mat --out model/model.joblib
```

This trains the ridge regression model with the same nested cross-validation as the notebook, logs params/metrics/the model artifact to a local MLflow run, and saves `model/model.joblib` for the API to load.

Inspect the run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000 — you'll see the run, its params
# (n_folds, test_size), and metrics (cv_mae_kcal_mol, test_mae_kcal_mol)
```

Every run is logged with its hyperparameters and metrics, and the model artifact is versioned alongside them.

## 2. Run the API locally (sanity check before containerizing)

```bash
uvicorn app.serve:app --reload --port 8000
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"H": 4, "C": 1}'
```

OR

```pwsh
curl http://localhost:8000/health
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"H": 4, "C": 1}'
```

## 3. Build the container image

```bash
docker build -t qm7-predictor:local .
docker run --rm -p 8000:8000 qm7-predictor:local
# curl the same endpoints as above to confirm the containerized version works
```

## 4. Create a local Kubernetes cluster and load the image

`kind` runs Kubernetes in Docker — no cloud account needed.

```bash
kind create cluster --name qm7-demo
kind load docker-image qm7-predictor:local --name qm7-demo
```


## 5. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods --watch   # wait for 2/2 Running
```

Port-forward to reach it:

```bash
kubectl port-forward svc/qm7-predictor 8000:80
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"H": 4, "C": 1}'
```

## 6. Demonstrate autoscaling

The HPA needs the metrics-server to read CPU usage, which kind doesn't ship by default:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# kind's metrics-server needs --kubelet-insecure-tls; patch it:
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
```

Then, with the port-forward from step 5 still running in one terminal:

```bash
# terminal 2
kubectl get hpa qm7-predictor --watch

# terminal 3
pip install requests
python loadtest.py --url http://localhost:8000/predict --requests 20000 --workers 50
```

Watch terminal 2 - you should see `TARGETS` CPU utilization climb above 50% and `REPLICAS` scale up from 2 toward the max of 6, then scale back down a few minutes after the load stops. 

## 7. Tear down

```bash
kind delete cluster --name qm7-demo
```

## What this demonstrates (mapped to common JD language)

- **Model training, evaluation, versioning** — `train.py` + MLflow run tracking (params, metrics, artifact)
- **Model serving via API** — FastAPI with a typed request/response schema
- **Containerization** — Dockerfile with a healthcheck
- **Deployment and orchestration** — Kubernetes Deployment, Service, and autoscaling via HPA, with readiness/liveness probes
- **Reproducibility** — pinned `requirements.txt`, deterministic training via a fixed seed
- **Model registry and governed promotion** — MLflow registry with a `champion` alias, gated so only non-regressing models reach serving
- **CI/CD for retraining** — GitHub Actions workflow that retrains, evaluates, and only builds/ships an image on genuine improvement
- **Hyperparameter search** — W&B Bayesian sweep over regularization strength, sharing code with the main training script

## 8. Model registry with gated promotion

Every `train.py` run now registers a model version and only promotes it to the `champion` alias if it doesn't regress the current champion's test MAE.


```bash
python app/train.py --data data/qm7.mat --out model/model.joblib
# -> Registered version N. Promoted: True/False (<reason>)
```

Inspect the registry:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Models tab -> qm7-atomization-energy -> see every version, its metrics,
# and which one carries the "champion" alias
```

Only the champion's artifact is copied to `model/champion.joblib`, which is what `serve.py` and the Dockerfile actually load - so a run that regresses performance gets recorded in the registry for visibility, but never reaches the serving path. For example: run training twice with different flags (e.g. `--test-size 0.6` starves the model of training data) and watch the second run get registered but rejected.

## 9. CI: retrain, gate, and ship only on improvement

`.github/workflows/retrain.yml` runs on a weekly schedule, on pushes that touch training code or data, and on manual dispatch. It:

1. Restores the MLflow registry from a GitHub Actions cache (so the champion comparison has history to compare against across runs)
2. Runs `train.py`, which registers a version and reports `promoted` as a step output
3. **Only if promoted** — builds the Docker image with the new champion baked in and pushes it to GHCR, tagged with the commit SHA

This is the gate that matters: most retraining runs on real data *won't* beat the current champion, and the workflow should do nothing in that case rather than ship a worse model. 

This uses a GitHub Actions cache as a stand-in for a real MLflow tracking server

## 10. Hyperparameter sweep with Weights & Biases

`app/sweep.py` replaces the manual `logspace` grid search in `train.py` with a Bayesian sweep over the regularization strength, reusing the same `build_features` / `cross_validate` functions so the two never drift apart.

```bash
pip install wandb
wandb login

cd app
wandb sweep sweep.yaml     # prints a sweep ID
wandb agent <sweep_id>     # launches runs; watch the dashboard fill in
```

The search space (`log_lambda` from -20 to 5) isn't arbitrary - the model is essentially flat until around `1e-3` and then degrades sharply past `lambda=1`, so there's a real minimum for the Bayesian search to find rather than a flat line it's exploring pointlessly.

## Possible further extensions

- Swap the atom-composition model for the notebook's pair-based model (Section 3.2) for a stronger MAE, at the cost of a heavier feature pipeline
- Wire the CI workflow's commented `kubectl set image` step to a real cluster, behind whatever review process fits
- Point `--mlflow-uri` at a hosted MLflow server instead of local SQLite, so the registry persists independent of any one machine or CI run
