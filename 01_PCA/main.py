import math
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Sets the resolution of the image
resolution = 200
total_pixels = resolution ** 2

# The folder name for the training data
path = "cropped_pics"

# Gets all the images by their name in the folder
labels = os.listdir(path)

# A list containing all the faces
face_vector = []

# For each face, reshape it into a vector and append it to the list
for label in labels:
    face_image = cv2.imread(os.path.join(path, label), cv2.IMREAD_GRAYSCALE)
    face_image = cv2.resize(face_image, (resolution, resolution))
    face_image = face_image.reshape(total_pixels,)
    face_vector.append(face_image)

face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()

# Calculate the average face
avg_face_vector = face_vector.mean(axis=1)

# Show the picture of the average face
plt.title("Average face")
plt.imshow(avg_face_vector.reshape(resolution, resolution), cmap="gray")
plt.show()

# Reshape the average face into a vector
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)

# Normalize each face in the face array by subtracting it from the average face
normalized_face_vector = face_vector - avg_face_vector

# Calculate the covariance matrix and the eigenvalues
covariance_matrix = np.cov(np.transpose(normalized_face_vector))
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Keep the k largest eigenvectors, and generate k eigenfaces
k = 10
k_eigen_vectors = eigen_vectors[0:k, :]
eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))

fig, axes = plt.subplots(math.floor(math.sqrt(len(eigen_faces))), math.floor(math.sqrt(len(eigen_faces)) + 1))
fig.suptitle("Eigenfaces decomposition")
index = 0
for i in range(len(axes)):
    for j in range(len(axes[i])):
        if index == len(eigen_faces) - 1:
            break
        pic = eigen_faces[index]
        pic = pic.reshape(resolution, resolution)
        axes[i][j].imshow(pic, cmap="gray")
        index += 1
plt.show()

# The name of the test image
test_add = "testing" + ".jpg"
test_img = cv2.imread(test_add)

# Load and resize the test image
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
test_img = cv2.resize(test_img, (resolution, resolution))
plt.title("Test face")
plt.imshow(test_img, cmap="gray")
plt.show()

# Reshape, rescale and normalize the test image
test_img = test_img.reshape(total_pixels, 1)
test_normalized_face_vector = test_img - avg_face_vector
test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
print(test_weight.shape)

# Find the index inside the train data that best matches the test data, and print the index
index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))
distance = np.linalg.norm(test_weight - weights, axis=1)
distance = distance / np.sum(distance)
print(np.around(distance, 3))
print(index, labels[index], np.round(distance[index], 3))
