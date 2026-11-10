import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

def loss_fn(out, y1, y2, y3, dataset):
    loss = 0.01 * criterion(out["level1"], y1)
    batch = y1.size(0)

    l2_loss_sum = 0.0
    l3_loss_sum = 0.0
    l3_count = 0

    for i in range(batch):
        l1_name = dataset.level1_classes[y1[i].item()]
        l2_name = dataset.level2_classes[y2[i].item()]

        level2_names = list(dataset.hierarchy[l1_name].keys())
        if l2_name not in level2_names:
            continue

        l2_local = level2_names.index(l2_name)
        l2_loss_sum += criterion(
            out["level2"][l1_name][i:i + 1],
            torch.tensor([l2_local], device=y1.device)
        )

        # Level 3 loss only applies if the branch has level 3 subfolders (e.g. Drones)
        level3_names = dataset.hierarchy[l1_name][l2_name]
        if len(level3_names) > 0:
            l3_name = dataset.level3_classes[y3[i].item()]
            if l3_name in level3_names:
                l3_local = level3_names.index(l3_name)
                l3_loss_sum += criterion(
                    out["level3"][f"{l1_name}_{l2_name}"][i:i + 1],
                    torch.tensor([l3_local], device=y1.device)
                )
                l3_count += 1

    total_loss = loss + (0.3 * l2_loss_sum / batch)

    if l3_count > 0:
        total_loss += (0.7 * l3_loss_sum / l3_count)

    return total_loss