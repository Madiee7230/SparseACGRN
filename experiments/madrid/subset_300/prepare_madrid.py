
from pathlib import Path
import json
import h5py
import numpy as np

root = Path(__file__).resolve().parents[3] / "datasets/madrid/subset_300"
data_dir = root / "raw/subset_300"
save_dir = root / "processed"
save_dir.mkdir(parents=True, exist_ok=True)

# Load traffic data
file = next(data_dir.rglob("MTD_training_seq_len12_horizon12.h5"))

with h5py.File(file, "r") as f:
    traffic = f["t/block0_values"][:].astype("float32")
    sensor_ids = f["t/axis0"][:]

# Load the original split indices
def load_split(name):
    path = file.parent / f"idx_{name}_MTD_training_seq_len12_horizon12.npy"
    return np.load(path)

train = load_split("train")
val = load_split("val")
test = load_split("test")

# Avoid overlapping prediction targets between splits
horizon = 12
train = train[train + horizon <= val[0]]
val = val[val + horizon <= test[0]]

# Normalize using training data only
train_end = int(train[-1] + horizon + 1)
mean = float(traffic[:train_end].mean())
std = float(traffic[:train_end].std())

if not np.isfinite(traffic).all() or std == 0:
    raise ValueError("Invalid traffic data")

traffic = (traffic - mean) / std

# Save files
np.save(save_dir / "traffic_normalized.npy", traffic)
np.save(save_dir / "sensor_ids.npy", sensor_ids)
np.savez(save_dir / "split_indices.npz",
         train=train, val=val, test=test)

metadata = {
    "mean": mean,
    "std": std,
    "num_sensors": len(sensor_ids),
    "history": 12,
    "horizon": horizon,
    "interval_minutes": 60
}

with open(save_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Traffic shape:", traffic.shape)
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Mean:", mean)
print("Std:", std)
print("Data preparation complete!")
