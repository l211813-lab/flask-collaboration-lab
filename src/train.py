import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_out", type=str, default="models/model.pkl")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv")).squeeze()

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    print("Model saved to", args.model_out)
