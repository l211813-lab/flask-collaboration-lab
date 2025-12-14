import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="metrics/eval.json")
    args = parser.parse_args()

    X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(args.data_dir, "y_test.csv")).squeeze()

    model = joblib.load(args.model)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds))
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics saved to", args.out)
    print(metrics)
