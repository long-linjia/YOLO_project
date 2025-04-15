import os
import cv2
import torch
from torch.utils.data import Dataset
from config import INPUT_SIZE, NUM_CLASSES

class COCODataset(Dataset):
    def __init__(self, data_dir, input_size=416, train=True):
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.input_size = input_size
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace(".jpg", ".txt"))

        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.transpose((2, 0, 1)) / 255.0
        img = torch.tensor(img, dtype=torch.float32)

        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    targets.append([cls, x, y, w, h])
        targets = torch.tensor(targets, dtype=torch.float32)

        return img, targets
