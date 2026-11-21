
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from experiments.madrid.subset_300.madrid_loader import MadridDataset

data_dir = Path(__file__).resolve().parents[3] / "datasets/madrid/subset_300/processed"

with open(data_dir / "metadata.json") as f:
    std = json.load(f)["std"]

for split in ["val", "test"]:
    loader = DataLoader(MadridDataset(split), batch_size=64)

    total_abs = 0
    total_sq = 0
    count = 0

    for x, y in loader:
        pred = x[:, -1:, :, :].expand_as(y)
        error = (pred - y) * std

        total_abs += error.abs().sum().item()
        total_sq += error.square().sum().item()
        count += error.numel()

    mae = total_abs / count
    rmse = (total_sq / count) ** 0.5

    print(f"{split.upper()} MAE: {mae:.2f}")
    print(f"{split.upper()} RMSE: {rmse:.2f}")
