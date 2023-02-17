import recognition
import cv2

def faceRecognitionPipeline(img):
    # Load face recogntion model
    haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')  # cascade classifier

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect all faces
    faces = haar.detectMultiScale(gray, 1.3, 5)
    predictions = []

    # For position of each face:
    for x, y, w, h in faces:
        # Crop the face
        roi = gray[y:y+h, x:x+w]
        
        # Predict the name of the face
        name, dist = recognition.predict(roi)
        text = f"{name} : {dist:.3f}"
        color = (255, 255, 0)

        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        output = {
            'roi': roi,
            'prediction_name': name,
            'score': dist
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
