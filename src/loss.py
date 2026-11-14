import torch
import torch.nn.functional as F
from src.config import FOCAL_GAMMA


def focal_loss_single(logits, target, gamma):
    """Computes focal loss for a set of logits and targets."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if target.dim() == 0:
        target = target.unsqueeze(0)

    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()

    focal_loss = -((1 - pt) ** gamma) * log_pt
    return focal_loss


def loss_fn(out, y1, y2, y3, dataset):
    device = y1.device
    batch = y1.size(0)

    # Level 1 Loss (computed across the batch)
    l1_loss_tensor = focal_loss_single(out["level1"], y1, gamma=FOCAL_GAMMA)
    mean_l1 = l1_loss_tensor.mean()

    l2_losses = []
    l3_losses = []

    for i in range(batch):
        l1_name = dataset.level1_classes[y1[i].item()]
        l2_name = dataset.level2_classes[y2[i].item()]

        level2_names = list(dataset.hierarchy[l1_name].keys())
        if l2_name not in level2_names:
            continue

        l2_local = level2_names.index(l2_name)
        l2_target = torch.tensor([l2_local], device=device)
        l2_logits = out["level2"][l1_name][i:i + 1]

        # Keep as tensor to preserve autograd gradients!
        l2_loss_val = focal_loss_single(l2_logits, l2_target, gamma=FOCAL_GAMMA)
        l2_losses.append(l2_loss_val)

        # Level 3 loss only applies if the branch has level 3 subfolders (e.g. Drones)
        level3_names = dataset.hierarchy[l1_name][l2_name]
        if len(level3_names) > 0:
            l3_name = dataset.level3_classes[y3[i].item()]
            if l3_name in level3_names:
                l3_local = level3_names.index(l3_name)
                l3_target = torch.tensor([l3_local], device=device)
                l3_logits = out["level3"][f"{l1_name}_{l2_name}"][i:i + 1]

                # Keep as tensor to preserve autograd gradients!
                l3_loss_val = focal_loss_single(l3_logits, l3_target, gamma=FOCAL_GAMMA)
                l3_losses.append(l3_loss_val)

    mean_l2 = torch.stack(l2_losses).mean() if len(l2_losses) > 0 else torch.tensor(0.0, device=device)
    mean_l3 = torch.stack(l3_losses).mean() if len(l3_losses) > 0 else torch.tensor(0.0, device=device)

    # Weights summing strictly to 1.0
    if len(l3_losses) > 0:
        # Level 1 (0.2) + Level 2 (0.3) + Level 3 (0.5) = 1.0
        total_loss = (0.2 * mean_l1) + (0.3 * mean_l2) + (0.5 * mean_l3)
    else:
        # Level 1 (0.4) + Level 2 (0.6) = 1.0 (when Level 3 isn't present)
        total_loss = (0.4 * mean_l1) + (0.6 * mean_l2)

    return total_loss