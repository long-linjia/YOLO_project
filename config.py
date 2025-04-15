DATA_PATH = "./data/coco/"
NUM_CLASSES = 80
INPUT_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)]
]
DEVICE = "cuda"