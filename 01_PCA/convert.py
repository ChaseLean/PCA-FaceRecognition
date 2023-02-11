import os
from PIL import Image

path = "train"
labels = os.listdir(path)
for label in labels:
    im = Image.open(os.path.join(path, label))
    rgb_im = im.convert("RGB")
    name = label.split(".")[0]
    rgb_im.save(f"{name}.jpg")