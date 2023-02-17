import os
import cv2
import processing
from pipeline import faceRecognitionPipeline

haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
pca = processing.load_data(os.path.join("models", "pca.pickle"))
model = processing.load_data(os.path.join("models", "svm.pickle"))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == False:
        break

    pred_img = faceRecognitionPipeline(frame, haar, pca, model)

    cv2.imshow('prediction', pred_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()