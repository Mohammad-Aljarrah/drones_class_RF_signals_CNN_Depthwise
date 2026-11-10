import os
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import EPOCHS, RESULT_DIR
from loss import loss_fn


def ddp_setup():
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except RuntimeError as e:
            if "gloo device" in str(e) or "unsupported gloo device" in str(e):
                print(
                    "Warning: Distributed process group init failed due to Windows network socket mapping. Falling back to single-device non-DDP mode.")
                return local_rank
            else:
                raise e

    torch.manual_seed(42)
    return local_rank


def reduce_tensor(x):
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


import pandas as pd


def save_cm(y_true, y_pred, classes, title, filename):
    if len(y_true) == 0:
        return

    # Compute confusion matrix and row-normalized percentages
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = (cm.astype(np.float32) / np.where(row_sums == 0, 1, row_sums)) * 100

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. Save the original heatmap image (.png) safely without text clutter
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_pct, annot=False, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    plt.close()

    # 2. Export detailed tables to Excel (.xlsx) with Raw Counts & Percentage sheets
    df_raw = pd.DataFrame(cm, index=classes, columns=classes)
    df_pct = pd.DataFrame(cm_pct, index=classes, columns=classes).round(2)

    excel_filename = filename.replace(".png", ".xlsx")
    excel_path = os.path.join(RESULT_DIR, excel_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw Counts')
        df_pct.to_excel(writer, sheet_name='Percentage (%)')

def evaluate(model, loader, device, dataset):
    model.eval()
    l1_true, l2_true, l3_true = [], [], []
    l1_pred, l2_pred, l3_pred = [], [], []

    with torch.no_grad():
        for batch in loader:
            # Handle potential dict vs tuple dataloader batch formats robustly
            if isinstance(batch, dict):
                x = batch['image']
                y1 = batch['level1']
                y2 = batch['level2']
                y3 = batch['level3']
            else:
                x, y1, y2, y3 = batch

            x = x.to(device)
            out = model(x)
            p1 = out["level1"].argmax(1)

            for i in range(x.size(0)):
                l1_name = dataset.level1_classes[p1[i].item()]
                l2_logits = out["level2"][l1_name][i]
                l2_local = l2_logits.argmax().item()

                level2_names = list(dataset.hierarchy[l1_name].keys())
                if l2_local >= len(level2_names):
                    continue
                l2_name = level2_names[l2_local]
                l2_global = dataset.level2_to_idx[l2_name]

                l1_pred.append(p1[i].cpu().item())
                l2_pred.append(l2_global)

                level3_names = dataset.hierarchy[l1_name][l2_name]
                if len(level3_names) > 0:
                    l3_logits = out["level3"][f"{l1_name}_{l2_name}"][i]
                    l3_local = l3_logits.argmax().item()
                    l3_name = level3_names[l3_local]
                    l3_global = dataset.level3_to_idx[l3_name]

                    l3_true.append(y3[i].cpu().item() if torch.is_tensor(y3) else y3[i])
                    l3_pred.append(l3_global)

            if torch.is_tensor(y1):
                l1_true.extend(y1.cpu().numpy())
                l2_true.extend(y2.cpu().numpy())
            else:
                l1_true.extend(y1)
                l2_true.extend(y2)

    # Print and save Level 1 (Domain)
    print("\nLEVEL 1 (Domain)")
    print(classification_report(l1_true, l1_pred, target_names=dataset.level1_classes, zero_division=0))
    save_cm(l1_true, l1_pred, dataset.level1_classes, "Level 1 Domain Confusion Matrix", "level1_cm.png")

    # Print and save Level 2 (Subclass)
    print("\nLEVEL 2 (Subclass)")
    print(classification_report(l2_true, l2_pred, target_names=dataset.level2_classes, zero_division=0))
    save_cm(l2_true, l2_pred, dataset.level2_classes, "Level 2 Subclass Confusion Matrix", "level2_cm.png")

    # Print and save Level 3 (Drone Status)
    if len(l3_true) > 0:
        print("\nLEVEL 3 (Drone Status)")
        print(classification_report(l3_true, l3_pred, target_names=dataset.level3_classes, zero_division=0))
        save_cm(l3_true, l3_pred, dataset.level3_classes, "Level 3 Status Confusion Matrix", "level3_cm.png")


def train(model, train_loader, val_loader, sampler, device, rank, dataset):
    for epoch in range(EPOCHS):
        if epoch <= 40:
            LRr = 8e-4
        elif epoch <= 60:
            LRr = 1e-4
        else:
            LRr = 1e-5

        optimizer = torch.optim.Adam(model.parameters(), lr=LRr)

        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        model.train()

        loss_sum = torch.tensor(0.0, device=device)

        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['image']
                y1 = batch['level1']
                y2 = batch['level2']
                y3 = batch['level3']
            else:
                x, y1, y2, y3 = batch

            x = x.to(device, non_blocking=True)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            out = model(x)
            loss = loss_fn(out, y1, y2, y3, dataset)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach()

        reduce_tensor(loss_sum)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            print(
                "Loss:",
                loss_sum.item() / (len(train_loader) * world_size)
            )
            print("Validation")

            eval_model = model.module if hasattr(model, "module") else model
            evaluate(eval_model, val_loader, device, dataset)

        if dist.is_initialized():
            dist.barrier()