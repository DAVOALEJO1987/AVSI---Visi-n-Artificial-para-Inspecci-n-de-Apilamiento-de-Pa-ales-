
from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import build_resnet18
from .data_processing import build_dataloaders

def train(
    split_dir: Path,
    out_models: Path = Path("models"),
    out_results: Path = Path("results/metrics"),
    img_size=(224,224),
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    patience=3,
    freeze_backbone=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_models.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, classes = build_dataloaders(split_dir, img_size, batch_size)
    model = build_resnet18(len(classes), pretrained=True, dropout=0.3, freeze_backbone=freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc, best_path, no_improve = 0.0, out_models/"best_model.pt", 0

    def _run_epoch(loader: DataLoader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train_mode:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train_mode):
                out = model(xb)
                loss = criterion(out, yb)
                if train_mode:
                    loss.backward(); optimizer.step()
            running_loss += loss.item() * xb.size(0)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)
        return running_loss/total, correct/total

    for ep in range(1, epochs+1):
        tr_loss, tr_acc = _run_epoch(train_loader, True)
        val_loss, val_acc = _run_epoch(val_loader, False)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        print(f"[{ep:02d}/{epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc; no_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": classes}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv(out_results/"history.csv", index=False)
    print("Mejor val_acc:", best_val_acc, "| modelo guardado en:", best_path)
    return best_path, classes

if __name__ == "__main__":
    train(split_dir=Path("data/processed"))
