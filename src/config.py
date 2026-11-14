import os

DATA_ROOT_Windows = r"C:\Users\k67192ma\Desktop\drones_class_RF_signals_CNN_Depthwise-master\drones_class_RF_signals_CNN_Depthwise-master\\datasets\full_dataset"
DATA_ROOT = "/home/k67192ma/data/drone_project"#set it to DATA_ROOT_Windows if you use Windows+CPU
MODEL_NAME = "se"
EPOCHS =60
BATCH = 6
IMG_SIZE = 224
FOCAL_GAMMA = 2.0  # focusing parameter of the Focal loss. Set to 0.0 to instantly fall back to standard cross-entropy
RESULT_DIR = r"C:\Users\k67192ma\Desktop\drones_class_RF_signals_CNN_Depthwise-master\drones_class_RF_signals_CNN_Depthwise-master\results"
os.makedirs(RESULT_DIR, exist_ok=True)