
import json
import torch
import torch.nn as nn

from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader

from experiments.madrid_loader import MadridDataset
from model.BaseAGCRN.AGCRN import AGCRN

torch.manual_seed(42)
torch.set_num_threads(4)

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available")

device = torch.device("cuda")
save_dir = Path("experiments/results/base_agcrn_64x2")
save_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path(__file__).resolve().parents[1] / "datasets/madrid/processed"
with open(data_dir / "metadata.json") as f:
    std = json.load(f)["std"]

train_loader = DataLoader(
    MadridDataset("train"), batch_size=16, shuffle=True
)
val_loader = DataLoader(
    MadridDataset("val"), batch_size=16
)
test_loader = DataLoader(
    MadridDataset("test"), batch_size=16
)

args = SimpleNamespace(
    num_nodes=300,
    input_dim=1,
    rnn_units=64,
    output_dim=1,
    horizon=12,
    num_layers=2,
    default_graph=False,
    embed_dim=10,
    cheb_k=2
)

model = AGCRN(args)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.L1Loss()


def train_epoch():
    model.train()
    total = 0
    count = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x, None)
        loss = criterion(pred, y)

        if not torch.isfinite(loss):
            raise ValueError("Invalid loss")

        loss.backward()
        optimizer.step()

        total += loss.item() * y.numel()
        count += y.numel()

    return total / count


def evaluate(loader):
    model.eval()
    total_abs = 0
    total_sq = 0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, None)

            error = (pred - y) * std
            total_abs += error.abs().sum().item()
            total_sq += error.square().sum().item()
            count += error.numel()

    mae = total_abs / count
    rmse = (total_sq / count) ** 0.5

    return mae, rmse


best_mae = float("inf")
patience = 10
no_improvement = 0
history = []

print("GPU:", torch.cuda.get_device_name(0))

for epoch in range(1, 81):
    train_loss = train_epoch()
    val_mae, val_rmse = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_mae": val_mae,
        "val_rmse": val_rmse
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.2f} | "
        f"Val RMSE: {val_rmse:.2f}",
        flush=True
    )

    if val_mae < best_mae:
        best_mae = val_mae
        no_improvement = 0

        torch.save(
            model.state_dict(),
            save_dir / "best_model.pth"
        )
    else:
        no_improvement += 1

    if no_improvement >= patience:
        print("Early stopping")
        break

with open(save_dir / "history.json", "w") as f:
    json.dump(history, f, indent=2)

model.load_state_dict(
    torch.load(save_dir / "best_model.pth", map_location=device)
)

test_mae, test_rmse = evaluate(test_loader)

print("\n=== TEST RESULTS ===")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print("Model saved to:", save_dir)
