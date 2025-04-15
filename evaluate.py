import time
import torch
from model.yolov4 import YOLOv4
from utils.dataset import COCODataset
from torch.utils.data import DataLoader
from config import *

def evaluate():
    model = YOLOv4(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("weights/yolov4.pth"))
    model.eval()

    dataset = COCODataset(DATA_PATH, INPUT_SIZE, train=False)
    dataloader = DataLoader(dataset, batch_size=1)

    total_time = 0
    with torch.no_grad():
        for i, (img, _) in enumerate(dataloader):
            img = img.to(DEVICE)
            start = time.time()
            _ = model(img)
            total_time += time.time() - start
    print(f"Avg FPS: {len(dataloader)/total_time:.2f}")