import cv2
from pipeline import faceRecognitionPipeline


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if ret == False:
        break

    pred_img, pred_dict = faceRecognitionPipeline(frame)

    cv2.imshow('prediction', pred_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()