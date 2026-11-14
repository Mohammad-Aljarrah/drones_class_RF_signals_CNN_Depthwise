import os
import cv2
import torch
from torch.utils.data import Dataset
from src.config import IMG_SIZE

import pandas as pd
from collections import Counter
from src.config import RESULT_DIR


def export_class_counts(dataset, filename="class_counts.xlsx"):
    """
    Counts total samples per class across Level 1, Level 2, and Level 3,
    and exports them into a multi-sheet Excel file.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    excel_path = os.path.join(RESULT_DIR, filename)

    # Collect ground-truth labels for all samples in the dataset
    # (Adapting to your dataset structure: level1, level2, level3 label lists)
    l1_counts = Counter([dataset.level1_classes[sample[1]] for sample in dataset.samples]) if hasattr(dataset, 'samples') else Counter()
    l2_counts = Counter([dataset.level2_classes[sample[2]] for sample in dataset.samples]) if hasattr(dataset, 'samples') else Counter()
    l3_counts = Counter([dataset.level3_classes[sample[3]] for sample in dataset.samples]) if hasattr(dataset, 'samples') else Counter()

    # Convert counters to DataFrames
    df_l1 = pd.DataFrame(list(l1_counts.items()), columns=["Level 1 Class", "Image Count"]).sort_values(by="Image Count", ascending=False)
    df_l2 = pd.DataFrame(list(l2_counts.items()), columns=["Level 2 Class", "Image Count"]).sort_values(by="Image Count", ascending=False)
    df_l3 = pd.DataFrame(list(l3_counts.items()), columns=["Level 3 Class", "Image Count"]).sort_values(by="Image Count", ascending=False)

    # Save each level to a separate tab in Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_l1.to_excel(writer, sheet_name='Level 1 (Domain)', index=False)
        df_l2.to_excel(writer, sheet_name='Level 2 (Subclass)', index=False)
        df_l3.to_excel(writer, sheet_name='Level 3 (Detail)', index=False)

    print(f"Exported class counts to: {excel_path}")


class DroneDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        if not os.path.exists(root):
            raise FileNotFoundError(root)

        self.level1_classes = ["Drones", "Terrestrial"]
        self.level1_to_idx = {c: i for i, c in enumerate(self.level1_classes)}

        self.hierarchy = {}
        level2 = set()
        level3 = {"N/A"}

        # ---------------------------------------------------------
        # DISCOVER ASYMMETRIC TREE (Sorted for deterministic DDP indexing)
        # ---------------------------------------------------------
        for l1 in sorted(os.listdir(root)):
            l1_path = os.path.join(root, l1)
            if not os.path.isdir(l1_path) or l1 not in self.level1_to_idx:
                continue

            self.hierarchy[l1] = {}

            for l2 in sorted(os.listdir(l1_path)):
                l2_path = os.path.join(l1_path, l2)
                if not os.path.isdir(l2_path):
                    continue

                level2.add(l2)
                self.hierarchy[l1][l2] = []

                sub_items = sorted(os.listdir(l2_path))
                has_l3 = any(os.path.isdir(os.path.join(l2_path, item)) and ("_FY" in item or "_HO" in item) for item in sub_items)

                if has_l3:
                    for l3 in sub_items:
                        l3_path = os.path.join(l2_path, l3)
                        if os.path.isdir(l3_path):
                            self.hierarchy[l1][l2].append(l3)
                            level3.add(l3)

        self.level2_classes = sorted(list(level2))
        self.level3_classes = sorted(list(level3))

        self.level2_to_idx = {c: i for i, c in enumerate(self.level2_classes)}
        self.level3_to_idx = {c: i for i, c in enumerate(self.level3_classes)}

        # -------------------------
        # NORMALIZATION
        # -------------------------
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # ---------------------------------------------------------
        # BUILD SAMPLES
        # ---------------------------------------------------------
        for l1 in self.hierarchy:
            l1_id = self.level1_to_idx[l1]

            for l2 in self.hierarchy[l1]:
                l2_id = self.level2_to_idx[l2]

                if l1 == "Drones":
                    # 3-level path parsing: Drones -> DroneType (l2) -> Status (l3) -> Images
                    for l3 in self.hierarchy[l1][l2]:
                        l3_id = self.level3_to_idx[l3]
                        folder = os.path.join(root, l1, l2, l3)
                        if not os.path.exists(folder):
                            continue

                        for f in sorted(os.listdir(folder)):
                            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                                self.samples.append((
                                    os.path.join(folder, f),
                                    l1_id,
                                    l2_id,
                                    l3_id
                                ))
                else:
                    # 2-level path parsing for Terrestrial: Terrestrial -> Type -> Images
                    folder = os.path.join(root, l1, l2)
                    if not os.path.exists(folder):
                        continue

                    dummy_l3_id = self.level3_to_idx["N/A"]
                    for f in sorted(os.listdir(folder)):
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.samples.append((
                                os.path.join(folder, f),
                                l1_id,
                                l2_id,
                                dummy_l3_id
                            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y1, y2, y3 = self.samples[idx]

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img = (img - self.mean) / self.std

        return (
            img,
            torch.tensor(y1, dtype=torch.long),
            torch.tensor(y2, dtype=torch.long),
            torch.tensor(y3, dtype=torch.long)
        )