# This code is to use a trained NN for inference (Testing/evaluation)
import torch
from torch.utils.data import random_split, DataLoader

from src.config import DATA_ROOT, MODEL_NAME, BATCH, RESULT_DIR
from src.dataset import DroneDataset
from src.models import build_model
from src.engine import evaluate


def run_inference():
    # 1. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize dataset
    print("Initializing dataset...")
    dataset = DroneDataset(DATA_ROOT)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. Recreate the exact same train/val/test splits used during training (seed 42)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    _, _, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # 4. Create test DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0
    )

    # 5. Build model architecture
    print(f"Building model: {MODEL_NAME}...")
    model = build_model(MODEL_NAME, dataset)

    # 6. Load saved weights
    model_path = "conditional_hierarchical_model.pth"
    print(f"Loading weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Could not find weight file '{model_path}'. Make sure you have trained the model at least once.")
        return

    model = model.to(device)

    # 7. Run evaluation on test split
    print("\nStarting Test Inference & Evaluation...")
    evaluate(model, test_loader, device, dataset)
    print(f"\nInference complete! Confusion matrix heatmaps saved to: {RESULT_DIR}")


if __name__ == "__main__":
    run_inference()