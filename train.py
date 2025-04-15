import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.yolov4 import YOLOv4
from utils.dataset import COCODataset
from utils.loss import YOLOLoss
from config import *

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def train():
    model = YOLOv4(NUM_CLASSES).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = YOLOLoss()

    dataset = COCODataset(DATA_PATH, INPUT_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    model.train()
    for epoch in range(EPOCHS):
        for images, targets in dataloader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")