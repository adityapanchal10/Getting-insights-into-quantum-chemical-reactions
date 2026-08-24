"""
Fires concurrent prediction requests so you can watch the HPA scale pods.

Usage:
    python loadtest.py --url http://localhost:8000/predict --requests 20000 --workers 50

Watch scaling in another terminal with:
    kubectl get hpa qm7-predictor --watch
    kubectl get pods --watch
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PAYLOAD = {"H": 8, "C": 3, "O": 1}


def fire(url):
    try:
        requests.post(url, json=PAYLOAD, timeout=5)
    except requests.RequestException:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fire, [args.url] * args.requests))
    elapsed = time.time() - start
    print(f"Sent {args.requests} requests with {args.workers} workers in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
