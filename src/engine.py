import os
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from src.config import EPOCHS, RESULT_DIR
from src.loss import loss_fn


def ddp_setup():
    # Safely initialize process group only if running via torchrun with CUDA available
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group(
            backend="gloo",
            init_method="env://"
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.manual_seed(42)
        return local_rank
    else:
        torch.manual_seed(42)
        return 0


def reduce_tensor(x):
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def save_cm(y_true, y_pred, classes, title, filename):
    if len(y_true) == 0 or len(classes) == 0:
        return

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = (cm.astype(np.float32) / np.where(row_sums == 0, 1, row_sums)) * 100

    os.makedirs(RESULT_DIR, exist_ok=True)

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

    df_raw = pd.DataFrame(cm, index=classes, columns=classes)
    df_pct = pd.DataFrame(cm_pct, index=classes, columns=classes).round(2)

    excel_filename = filename.replace(".png", ".xlsx")
    excel_path = os.path.join(RESULT_DIR, excel_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw Counts')
        df_pct.to_excel(writer, sheet_name='Percentage (%)')


def evaluate(model, val_loader, device, dataset):
    model.eval()

    val_l1_preds, val_l1_true = [], []
    val_l2_preds, val_l2_true = [], []
    val_l3_preds, val_l3_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch['image']
                y1 = batch['level1']
                y2 = batch['level2']
                y3 = batch['level3']
            else:
                x, y1, y2, y3 = batch

            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            out = model(x)
            batch_size = x.size(0)

            for i in range(batch_size):
                # Level 1 Evaluation
                l1_logits = out["level1"][i:i + 1]
                l1_pred = l1_logits.argmax(1).item()
                val_l1_preds.append(l1_pred)
                val_l1_true.append(y1[i].item())

                # Level 2 Evaluation (Mapping local branch predictions back to global indices)
                l1_name = dataset.level1_classes[y1[i].item()]
                l2_name = dataset.level2_classes[y2[i].item()]
                level2_names = list(dataset.hierarchy[l1_name].keys())

                if l2_name in level2_names:
                    l2_logits = out["level2"][l1_name][i:i + 1]
                    l2_pred_local = l2_logits.argmax(1).item()

                    predicted_l2_name = level2_names[l2_pred_local]
                    if predicted_l2_name in dataset.level2_classes:
                        global_pred_idx = dataset.level2_classes.index(predicted_l2_name)
                        val_l2_preds.append(global_pred_idx)
                        val_l2_true.append(y2[i].item())

                # Level 3 Evaluation (Drones branch local binary sub-branches: _FY vs _HO)
                level3_names = dataset.hierarchy[l1_name].get(l2_name, [])
                if len(level3_names) > 0:
                    l3_name = dataset.level3_classes[y3[i].item()]
                    if l3_name in level3_names:
                        l3_logits = out["level3"][f"{l1_name}_{l2_name}"][i:i + 1]
                        l3_pred_local = l3_logits.argmax(1).item()
                        l3_true_local = level3_names.index(l3_name)

                        val_l3_preds.append(l3_pred_local)
                        val_l3_true.append(l3_true_local)

    print("\n" + "=" * 40)
    print("VALIDATION REPORT")
    print("=" * 40)

    print("\nLEVEL 1")
    print(classification_report(
        val_l1_true, val_l1_preds,
        target_names=getattr(dataset, 'level1_classes', None),
        zero_division=0
    ))

    print("\nLEVEL 2")
    print(classification_report(
        val_l2_true, val_l2_preds,
        target_names=getattr(dataset, 'level2_classes', None),
        zero_division=0
    ))

    if len(val_l3_true) > 0:
        print("\nLEVEL 3 (Drones Sub-branches: _FY vs _HO)")
        print(classification_report(
            val_l3_true, val_l3_preds,
            target_names=["_FY", "_HO"],
            zero_division=0
        ))

    # Save Confusion Matrices and Excel Reports for all levels
    save_cm(
        val_l1_true, val_l1_preds,
        getattr(dataset, 'level1_classes', [str(i) for i in range(len(set(val_l1_true)))]),
        "Level 1 CM",
        "val_level1_cm.png"
    )

    save_cm(
        val_l2_true, val_l2_preds,
        getattr(dataset, 'level2_classes', [str(i) for i in range(len(set(val_l2_true)))]),
        "Level 2 CM",
        "val_level2_cm.png"
    )

    if len(val_l3_true) > 0:
        save_cm(
            val_l3_true, val_l3_preds,
            ["_FY", "_HO"],
            "Level 3 CM",
            "val_level3_cm.png"
        )


def train(model, train_loader, val_loader, sampler, device, rank, dataset):
    for epoch in range(EPOCHS):
        if epoch <= 40:
            LRr = 8e-3
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

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            print("Loss:", loss_sum.item() / (len(train_loader) * world_size))
            print("Validation")

            eval_model = model.module if hasattr(model, "module") else model
            evaluate(eval_model, val_loader, device, dataset)

        if dist.is_initialized():
            dist.barrier()