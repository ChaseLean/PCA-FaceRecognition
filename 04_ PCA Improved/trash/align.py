import cv2
import numpy as np
from scipy.spatial.distance import euclidean


def align_face(img, face_detector, resolution):
    eye_pos_w, eye_pos_h = 0.28, 0.28
    width, height = resolution, resolution
    img_rgb = img[..., ::-1]

    results = face_detector.detect_faces(img_rgb)
    result = results[0]

    l_e = result['keypoints']['left_eye']
    r_e = result['keypoints']['right_eye']
    center = (((r_e[0] + l_e[0]) // 2), ((r_e[1] + l_e[1]) // 2))

    dx = (r_e[0] - l_e[0])
    dy = (r_e[1] - l_e[1])
    dist = euclidean(l_e, r_e)

    angle = np.degrees(np.arctan2(dy, dx)) + 360
    scale = width * (1 - (2 * eye_pos_w)) / dist

    tx = width * 0.5
    ty = height * eye_pos_h

    m = cv2.getRotationMatrix2D(center, angle, scale)

    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])

    face_align = cv2.warpAffine(img, m, (width, height))

    return face_align

from matplotlib import pyplot as plt
import mtcnn
img = cv2.imread("pic.jpeg")
plt.imshow(img, cmap="gray")
img = align_face(img, mtcnn.MTCNN(), resolution=200)
plt.imshow(img, cmap="gray")
plt.show()
