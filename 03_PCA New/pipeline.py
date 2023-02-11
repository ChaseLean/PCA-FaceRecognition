import numpy as np
import pandas as pd
import sklearn
import pickle
import recognition
import math
import matplotlib.pyplot as plt
import cv2

def faceRecognitionPipeline(img):
    # Load all models
    haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')  # cascade classifier

    # Create Pipeline
    # step-01: read image
    # img = cv2.imread('./test_images/03.jpg')  # BGR
    # step-02: convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # step-03: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    predictions = []
    for x, y, w, h in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = gray[y:y+h, x:x+w]
        if roi.shape[1] > 200:
            roi_resize = cv2.resize(roi, (200, 200), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (200, 200), cv2.INTER_CUBIC)
        
        name, confidence = recognition.predict(roi)
        text = f"{name} : {confidence:.0f}%"
        color = (255, 255, 0)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        output = {
            'roi': roi,
            'prediction_name': name,
            'score': confidence
        }

        predictions.append(output)
    return img, predictions

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10, 10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()

# # generate report
# for i in range(len(predictions)):
#     print('Predicted Name =', predictions[i]['prediction_name'])
#     print('Predicted score = {:,.2f} %'.format(predictions[i]['score']))

#     print('-'*100)
