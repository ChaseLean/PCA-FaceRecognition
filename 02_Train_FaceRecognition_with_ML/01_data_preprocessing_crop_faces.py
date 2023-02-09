import numpy as np  # numerical python
import cv2  # opencv
import os as os
from glob import glob  # glob use to extract path of file
import matplotlib.pyplot as plt  # visualze and display

exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.svg']

# this will return all images path in a list
# fpath = [f for ext in exts for f in glob(
#     os.path.join("./data/female/", ext))]
# mpath = [f for ext in exts for f in glob(
#     os.path.join('./data/male/', ext))]
rpath = [f for ext in exts for f in glob(
    os.path.join('./data/raw/', ext))]
# print('The number of images in Female folder = ', len(fpath))
# print('The number of images in Male folder = ', len(mpath))
print('The number of images in Raw folder = ', len(rpath))

# Step -1 Read Image and Convert to RGB
img = cv2.imread(rpath[0])  # read image in BGR
# this step will convert image from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step -2: Apply Haar Cascade Classifier
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
faces_list = haar.detectMultiScale(gray, 1.5, 5)
print(faces_list)
for x, y, w, h in faces_list:
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Step -3: crop Face
    roi = img_rgb[y:y+h, x:x+w]

    plt.imshow(roi)
    plt.axis('off')
    plt.show()


# Step -4: Save Image
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# Looping to crop all female and male images
# Crop female faces
# for i in range(len(fpath)):
#     try:

#         # Step -1 Read Image and Convert to RGB
#         img = cv2.imread(fpath[i])  # read image in BGR
#         # this step will convert image from BGR to RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Step -2: Apply Haar Cascade Classifier
#         gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         faces_list = haar.detectMultiScale(gray, 1.5, 5)
#         for x, y, w, h in faces_list:
#             # Step -3: crop Face
#             roi = img[y:y+h, x:x+w]
#             # Step -4: Save Image
#             cv2.imwrite(f'./crop_data/female/female_{i}.jpg', roi)
#             print('Image Sucessfully processed')
#     except:
#         print('Unable to Process the image')

# # Crop male faces
# for i in range(len(mpath)):
#     try:

#         # Step -1 Read Image and Convert to RGB
#         img = cv2.imread(mpath[i])  # read image in BGR
#         # this step will convert image from BGR to RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Step -2: Apply Haar Cascade Classifier
#         gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         faces_list = haar.detectMultiScale(gray, 1.5, 5)
#         for x, y, w, h in faces_list:
#             # Step -3: crop Face
#             roi = img[y:y+h, x:x+w]
#             # Step -4: Save Image
#             cv2.imwrite(f'./crop_data/male/male_{i}.jpg', roi)
#             print('Image Sucessfully processed')
#     except:
#         print('Unable to Process the image')

# Crop raw faces
for i in range(len(rpath)):
    try:

        # Step -1 Read Image and Convert to RGB
        img = cv2.imread(rpath[i])  # read image in BGR
        # this step will convert image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step -2: Apply Haar Cascade Classifier
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        faces_list = haar.detectMultiScale(gray, 1.5, 5)
        for x, y, w, h in faces_list:
            # Step -3: crop Face
            roi = img[y:y+h, x:x+w]
            # Step -4: Save Image
            cv2.imwrite(f'./crop_data/raw/raw_{i}.jpg', roi)
            print('Image Sucessfully processed')
    except:
        print('Unable to Process the image')
