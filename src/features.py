import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["features"]

    df = pd.read_csv(args.in_csv)

    y = df["price"]
    X = df.drop(columns=["price"])

    # One-hot encoding
    X = pd.get_dummies(X, columns=["property_type", "city", "province", "location"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    os.makedirs(args.out_dir, exist_ok=True)

    X_train.to_csv(os.path.join(args.out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(args.out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(args.out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(args.out_dir, "y_test.csv"), index=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump(list(X.columns), "models/feature_columns.pkl")

    print("Train/test CSVs saved in", args.out_dir)
