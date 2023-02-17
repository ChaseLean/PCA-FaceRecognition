# https://stackoverflow.com/questions/59104545/using-mtcnn-with-a-webcam-via-opencv

import os
import cv2
import numpy as np
import processing
from mtcnn import MTCNN
from scipy.spatial.distance import euclidean
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

font = cv2.FONT_HERSHEY_SIMPLEX

resolution = 100

def align(face, color):
    width, height = resolution, resolution
    eye_pos_w, eye_pos_h = 0.28, 0.28
    l_e = face['keypoints']['left_eye']
    r_e = face['keypoints']['right_eye']
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

    face_align = cv2.warpAffine(color, m, (width, height))

    return face_align

def find_face_MTCNN(color, result_list, pca, model):
    for result in result_list:
        x, y, w, h = result['box']
        roi = align(result, color)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = processing.correct_lighting(roi)
        cv2.imwrite("vid_capture.jpg", roi)
        
        # Predict the name of the face
        name = model.predict(pca.transform([roi.flatten()]))[0]
        text = str(name)
        cv2.rectangle(color, (x, y), (x+w, y+h), (255, 255, 0), 3)
        # Draw a rectangle around the face
        cv2.putText(color, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 0), 3)
    return color


video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = MTCNN(min_face_size=80)
pca = processing.load_data(os.path.join("models", "pca.pickle"))
model = processing.load_data(os.path.join("models", "svm.pickle"))

while True:
    _, color = video_capture.read()
    faces = detector.detect_faces(color)
    detectFaceMTCNN = find_face_MTCNN(color, faces, pca, model)
    cv2.imshow('Video', detectFaceMTCNN)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()