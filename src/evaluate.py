
from __future__ import annotations
from pathlib import Path
import json, time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, average_precision_score
import matplotlib.pyplot as plt

def evaluate(
    processed_dir: Path = Path("data/processed"),
    model_path: Path = Path("models/best_model.pt"),
    out_results: Path = Path("results/metrics"),
    out_figs: Path = Path("results/figures"),
    img_size=(224,224),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_ds = datasets.ImageFolder(str(Path(processed_dir)/"test"), transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt.get("classes", test_ds.classes)
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = torch.nn.Linear(in_feats, len(classes))
    state = ckpt.get("model_state", ckpt.get("state_dict"))
    model.load_state_dict(state)
    model.eval().to(device)

    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_scores.append(probs)
            y_pred.extend(probs.argmax(1).tolist())
            y_true.extend(yb.numpy().tolist())

    y_scores = np.vstack(y_scores)
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_results/"confusion_matrix.csv")

    # mAP macro
    aps = []
    y_true_arr = np.array(y_true)
    for i in range(len(classes)):
        bin_true = (y_true_arr==i).astype(int)
        aps.append(average_precision_score(bin_true, y_scores[:,i]))
    mAP = float(np.nanmean(aps))

    # IoU por clase desde CM
    IoUs = []
    for i in range(len(classes)):
        TP = cm[i,i]; FP = cm[:,i].sum()-TP; FN = cm[i,:].sum()-TP
        denom = TP+FP+FN
        IoUs.append(float(TP/denom) if denom>0 else float("nan"))
    mIoU = float(np.nanmean(IoUs))

    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    pd.DataFrame(rep).to_csv(out_results/"classification_report.csv")

    with open(out_results/"summary.json","w") as f:
        json.dump({"accuracy": acc, "mAP": mAP, "mIoU": mIoU, "classes": classes}, f, indent=2)

    # Simple plot CM
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión (test)")
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.colorbar()
    plt.savefig(out_figs/"confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # FPS estimado
    N_images, t_total = 0, 0.0
    with torch.no_grad():
        t0 = time.time()
        for xb, _ in test_loader:
            xb = xb.to(device); _ = model(xb)
            N_images += xb.size(0)
        t_total = time.time() - t0
    fps = N_images / max(t_total, 1e-9)

    return {"accuracy": acc, "mAP": mAP, "mIoU": mIoU, "fps": fps, "classes": classes}

if __name__ == "__main__":
    print(evaluate())
