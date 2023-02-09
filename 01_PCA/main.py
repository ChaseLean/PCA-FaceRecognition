import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

resolution = 512
pca = PCA(200)

image_data = []

for i in range(2):
    img = cv2.imread(f"pic{i}.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (resolution, resolution))
    image_data.append(img)

avg_face = np.zeros_like(image_data[0])

for i in range(len(image_data)):
    avg_face += image_data[i]
avg_face = avg_face / len(image_data)

transformed = pca.fit_transform(avg_face)
inverted = pca.inverse_transform(transformed)
img_compressed = inverted.astype(np.uint8)

plt.imshow(img_compressed, cmap="gray")
plt.show()