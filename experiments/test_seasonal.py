
import json
import numpy as np
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1] / "datasets/madrid/processed"

traffic = np.load(data_dir / "traffic_normalized.npy")

with np.load(data_dir / "split_indices.npz") as f:
    splits = {name: f[name] for name in ["val", "test"]}

with open(data_dir / "metadata.json") as f:
    std = json.load(f)["std"]


def evaluate(indices, lag):
    hours = indices[:, None] + np.arange(1, 13)
    actual = traffic[hours]
    predicted = traffic[hours - lag]

    error = (predicted.astype("float64") - actual) * std

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return mae, rmse


for split, indices in splits.items():
    print(f"\n{split.upper()}")

    for name, lag in [("Daily", 24), ("Weekly", 168)]:
        mae, rmse = evaluate(indices, lag)
        print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
