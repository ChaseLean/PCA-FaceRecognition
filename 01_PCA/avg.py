import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import os

resolution = 50
total_pixels = resolution ** 2
path = "Pics"
labels = os.listdir(path)

face_vector = []
for label in labels:
    face_image = cv2.cvtColor(cv2.imread(os.path.join(path, label)), cv2.COLOR_RGB2GRAY)
    face_image = cv2.resize(face_image, (resolution, resolution))
    face_image = face_image.reshape(total_pixels,)
    face_vector.append(face_image)

face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()

avg_face_vector = face_vector.mean(axis=1)
plt.title("Average face")
plt.imshow(avg_face_vector.reshape(resolution, resolution), cmap="gray")
plt.show()

avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector

covariance_matrix = np.cov(np.transpose(normalized_face_vector))
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

k = 10
k_eigen_vectors = eigen_vectors[0:k, :]
eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))

k = eigen_faces.shape[0]
fig, axes = plt.subplots(math.floor(math.sqrt(k)), math.floor(math.sqrt(k) + 1))
fig.suptitle("Eigenfaces decomposition")
index = 0
for i in range(len(axes)):
    for j in range(len(axes[i])):
        pic = eigen_faces[index]
        pic = pic.reshape(resolution, resolution)
        axes[i][j].imshow(pic, cmap="gray")
        index += 1
        if index > k:
            break
plt.show()






test_add = "0" + ".jpg"
test_img = cv2.imread(test_add)
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
test_img = cv2.resize(test_img, (resolution, resolution))
plt.title("Test face")
plt.imshow(test_img, cmap="gray")
plt.show()

test_img = test_img.reshape(total_pixels, 1)
test_normalized_face_vector = test_img - avg_face_vector
test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))

index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))
print(index)
