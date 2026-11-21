
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

data_dir = Path(__file__).resolve().parents[3] / "datasets/madrid/subset_300/processed"


class MadridDataset(Dataset):
    def __init__(self, split):
        self.traffic = np.load(
            data_dir / "traffic_normalized.npy",
            mmap_mode="r"
        )

        with np.load(data_dir / "split_indices.npz") as splits:
            self.indices = splits[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t = int(self.indices[index])

        x = self.traffic[t-11:t+1, :, None].copy()
        y = self.traffic[t+1:t+13, :, None].copy()

        return torch.from_numpy(x), torch.from_numpy(y)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dataset = MadridDataset(split)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        x, y = next(iter(loader))

        print(f"\n{split.upper()}")
        print("Samples:", len(dataset))
        print("Input shape:", x.shape)
        print("Target shape:", y.shape)
        print("Input finite:", torch.isfinite(x).all().item())
        print("Target finite:", torch.isfinite(y).all().item())
