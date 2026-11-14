# Conditional Hierarchical Radio Signal Classification Through Their Spectrograms

A multi-level CNN-based deep learning framework built with PyTorch to classify spectrogram datasets according to their hierarchy. The code is scalable, it supports seamless multi-GPU execution via `DistributedDataParallel` (DDP) on local workstations as well as institutional High-Performance Computing (HPC) clusters (i.e., CSF3 at the University of Manchester) using Slurm. ---

## 📋 Overview & Architecture

* **Dataset:** The dataset is collected from 2 datasets available online. The terrestrial subset is downloaded from here https://www.kaggle.com/datasets/halcy0nic/radio-frequecy-rf-signal-image-classification, while the Drone subset is downloaded from IEEE https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine. The total number of spectrograms used for training and validation is ~18,000+.
* **Hierarchical Multi-Branch Output:** Simultaneously predicts and evaluates across multiple classification levels:
  The hierarchy in the dataset is as follows
```text
   ├── Terrestrial signals (e.g., General Objects/Classes)
   │   ├── Level 2 (Types)
   │   │   ├── Wi-Fi
   │   │   ├── Cellular
   │   │   ├── Bluetooth
   │   │   ├── FM
   │   │   └── ...
   │
   ├── Drones signals
   │   ├── Level 2 (Drone Types)
   │   │   ├── AIR
   │   │   ├── DIS
   │   │   ├── INS
   │   │   ├── PHA
   │   │   └── ...
   │    └── Level 3 (Drone status)
   │        ├── _FY (Flying)
   │        └── _HO (Hovering)
```
* **Dynamic Learning Rate Scheduling:** Learning rate decreases across training epochs for stable convergence.
* **Distributed Training (`DDP`):** Uses `torchrun` and process group synchronization (`all_reduce`, `barrier`) to scale across any available GPU count without modifying core Python logic.

---

## 📂 Repository Structure

```text
├── src/
│   ├── config.py         # Global hyperparameters data paths
│   ├── dataset.py        # Dataset loading while considering inherent hierarchies
│   ├── engine.py         # core distributed training loop, multi-level hierarchical evaluation, and performance reporting
│   ├── loss.py           # Custom multi-level hierarchical loss function
│   └── model.py          # Multi-branch neural network architecture
├── main.py               # Main entry point: DDP setup, training loop, and evaluation
├── inference_only.py     # Inference code using trained and saved model, inference time, inference confusion matrix
├── run_training.sh       # Slurm batch script for HPC cluster job submission
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Environment Setup

### For GPU training, the code was tested on WSL, linux distribution on Windows

### 1. Clone the Repository
```bash
git clone https://github.com/Mohammad-Aljarrah/drones_class_RF_signals_CNN_Depthwise
cd </drones_class_RF_signals_CNN_Depthwise>
```

### 2. Create and Activate Virtual Environment (WSL)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Copy dataset to Linux distribution
```bash
mkdir -p ~/data/drone_project
cp -r /mnt/c/Users/k67192ma/Desktop/drones_class_RF_signals_CNN_Depthwise-master/drones_class_RF_signals_CNN_Depthwise-master/datasets/full_dataset/* ~/data/drone_project/

```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Ensure PyTorch is installed with the appropriate CUDA version matching your local drivers or cluster modules.)*

---

## 🚀 Usage

### Running Locally (Multi-GPU)
To train across multiple GPUs on a local machine (e.g., 3 GPUs), use `torchrun`:
```bash
torchrun --nproc-per-node=3 main.py
```
*(Adjust `--nproc-per-node` to match the number of GPUs available on your workstation. Note that batch size in `config.py` is per-GPU.)*

### Running on an HPC Cluster (CSF3 / Slurm)
1. **Configure Storage:** Place your codebase in your home directory (`~`) and point `RESULT_DIR` and dataset paths in `src/config.py` to high-speed scratch storage (`~/scratch`).
2. **Submit the Job:**
   ```bash
   sbatch run_training.sh
   ```
3. **Monitor Progress:**
   ```bash
   # Check queue status
   squeue -u $USER

   # Watch live training output and validation metrics
   tail -f drone_train_<job_id>.log
   ```
4. **Retrieve Results:** Copy generated reports back to your local machine via SCP or an SFTP client (e.g., MobaXterm / FileZilla):
   ```bash
   scp -r <username>@csf3.itservices.manchester.ac.uk:~/scratch/<project_folder>/results ./local_results
   ```

---

## 📊 Evaluation & Outputs

At the end of each epoch, Rank 0 runs validation and outputs structured performance diagnostics to `RESULT_DIR`:

* **Console Classification Reports:** Precision, recall, and F1-scores via `scikit-learn` for Level 1, Level 2, and Level 3 branches.
* **Confusion Matrix Heatmaps (`.png`):** High-resolution (300 DPI) percentage heatmaps (`val_level1_cm.png`, `val_level2_cm.png`, `val_level3_cm.png`).
* **Excel Reports (`.xlsx`):** Multi-sheet workbooks (`val_level1_cm.xlsx`, etc.) containing both **Raw Counts** and **Percentage (%)** matrices for downstream analysis.