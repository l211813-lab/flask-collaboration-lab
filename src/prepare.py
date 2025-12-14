import argparse
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default="data/zameen-updated.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = [
        "price",
        "property_type",
        "city",
        "province_name",
        "location",
        "latitude",
        "longitude"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {df.columns.tolist()}")

    df = df[required].copy()

    # Rename for consistency
    df.rename(columns={"province_name": "province"}, inplace=True)

    # Clean price
    df["price"] = df["price"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df.dropna(inplace=True)

    out_path = os.path.join(args.out_dir, "cleaned.csv")
    df.to_csv(out_path, index=False)

    print("Saved cleaned.csv to", out_path)
