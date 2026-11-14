
# Inference script for trained hierarchical drone classification model
# Measures pure neural network processing time.

import time
import torch
from torch.utils.data import random_split, DataLoader

from src.config import DATA_ROOT_Windows, MODEL_NAME, BATCH, RESULT_DIR
from src.dataset import DroneDataset
from src.models import build_model
from src.engine import evaluate


def run_inference():

    # 1. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize dataset
    print("\nInitializing dataset...")
    dataset = DroneDataset(DATA_ROOT_Windows)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. Recreate original train/validation/test splits
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

    # 5. Build model
    print(f"\nBuilding model: {MODEL_NAME}...")
    model = build_model(MODEL_NAME, dataset)

    # 6. Load trained weights
    model_path = "conditional_hierarchical_model.pth"

    print(f"Loading weights from {model_path}...")

    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(state_dict)

    except FileNotFoundError:
        print(f"Error: Weight file '{model_path}' not found.")
        return

    model = model.to(device)
    model.eval()

    # 7. Warm-up (excluded from timing)
    print("\nWarming up model...")

    with torch.inference_mode():

        for i, batch in enumerate(test_loader):

            if isinstance(batch, dict):
                inputs = batch["image"].to(device)
            else:
                inputs = batch[0].to(device)

            _ = model(inputs)

            if i >= 2:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 8. Measure pure neural network inference time
    print("\nStarting inference latency measurement...")

    total_inference_time = 0.0
    total_samples = 0

    with torch.inference_mode():

        for batch in test_loader:

            # Extract spectrogram input
            if isinstance(batch, dict):
                inputs = batch["image"].to(device)
            else:
                inputs = batch[0].to(device)

            # Synchronize GPU before timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # Complete model forward pass
            outputs = model(inputs)

            # Hierarchical predictions
            # Level 1: Domain prediction
            level1_predictions = outputs["level1"].argmax(dim=1)

            # Level 2: Predictions from all conditional heads
            level2_predictions = {
                key: logits.argmax(dim=1)
                for key, logits in outputs["level2"].items()
            }

            # Level 3: Predictions from all available status heads
            level3_predictions = {
                key: logits.argmax(dim=1)
                for key, logits in outputs["level3"].items()
            }

            # Synchronize GPU after timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Accumulate processing time
            total_inference_time += end_time - start_time
            total_samples += inputs.size(0)

    # 9. Calculate inference statistics
    if total_samples > 0:

        avg_latency = total_inference_time / total_samples

        throughput = (
            total_samples / total_inference_time
            if total_inference_time > 0 else 0
        )

        avg_batch_time = (
            total_inference_time / len(test_loader)
            if len(test_loader) > 0 else 0
        )

        print("\n" + "=" * 55)
        print("NEURAL NETWORK INFERENCE TIME RESULTS")
        print("=" * 55)

        print(f"Device                       : {device}")
        print(f"Total test samples           : {total_samples}")
        print(f"Batch size                   : {BATCH}")
        print(f"Total batches                : {len(test_loader)}")
        print(f"Total NN processing time     : {total_inference_time:.6f} seconds")
        print(f"Average batch processing     : {avg_batch_time * 1000:.4f} ms")
        print(f"Average inference latency    : {avg_latency * 1000:.4f} ms/sample")
        print(f"Inference throughput         : {throughput:.2f} samples/second")

        print("=" * 55)

    else:
        print("No test samples available.")

    # 10. Run original evaluation separately
    # Evaluation time is NOT included in inference timing.
    print("\nStarting original model evaluation...")

    evaluate(model, test_loader, device, dataset)

    print("\nInference and evaluation complete!")
    print(f"Confusion matrices saved to: {RESULT_DIR}")


if __name__ == "__main__":
    run_inference()
