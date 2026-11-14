# cd /mnt/c/Users/k67192ma/Desktop/drones_class_RF_signals_CNN_Depthwise-master/drones_class_RF_signals_CNN_Depthwise-master
#python3 -m venv venv
#source venv/bin/activate
#mkdir -p ~/data/drone_project
#rm -rf ~/data/drone_project
#cp -r /mnt/c/Users/k67192ma/Desktop/drones_class_RF_signals_CNN_Depthwise-master/drones_class_RF_signals_CNN_Depthwise-master/datasets/full_dataset/* ~/data/drone_project/

#pip install -r requirements.txt
#torchrun --nproc-per-node=3 main.py
#python main.py (For CPU training)
# pkill -f main.py
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import DATA_ROOT, MODEL_NAME, BATCH, RESULT_DIR
from src.dataset import DroneDataset
from src.models import build_model
from src.engine import ddp_setup, train, evaluate

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    rank = ddp_setup()
    is_distributed = dist.is_initialized()

    # Robust device selector: supports multi-GPU DDP, single GPU, and CPU fallback
    if torch.cuda.is_available() and is_distributed:
        device = torch.device(f"cuda:{rank}")
        print(f"Using DDP GPU device: cuda:{rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using Single GPU device: cuda:0")
    else:
        device = torch.device("cpu")
        print("No GPU found. Running training on CPU.")

    print("Initializing dataset...")
    dataset = DroneDataset(DATA_ROOT)
    print(f"Dataset loaded with {len(dataset)} samples.")

    from src.dataset import export_class_counts
    export_class_counts(dataset)

    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    if is_distributed:
        sampler = DistributedSampler(train_ds, shuffle=True)
    else:
        sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0
    )

    model = build_model(MODEL_NAME, dataset)
    model = model.to(device)

    if is_distributed and device.type == "cuda":
        model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=True
        )

    print("Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        sampler,
        device,
        rank,
        dataset
    )

    if is_distributed:
        dist.barrier()

    if rank == 0:
        save_target = model.module if hasattr(model, "module") else model
        torch.save(
            save_target.state_dict(),
            "conditional_hierarchical_model.pth"
        )
        print("\nFINAL TEST EVALUATION")
        evaluate(
            save_target,
            val_loader,
            device,
            dataset
        )

    if is_distributed:
        dist.destroy_process_group()