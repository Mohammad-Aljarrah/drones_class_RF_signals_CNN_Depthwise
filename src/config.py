import os

DATA_ROOT = r"C:\Users\k67192ma\PycharmProjects\RF_Drone_CNN_Depthwise\datasets\full_dataset"

MODEL_NAME = "se"
EPOCHS = 1
BATCH = 2
IMG_SIZE = 224

RESULT_DIR = r"C:\Users\k67192ma\PycharmProjects\RF_Drone_CNN_Depthwise\results"
os.makedirs(RESULT_DIR, exist_ok=True)