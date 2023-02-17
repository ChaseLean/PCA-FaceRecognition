import cv2
import processing

resolution = 100

def faceRecognitionPipeline(img, haar, pca, model):

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect all faces
    faces = haar.detectMultiScale(gray, 1.3, 5)

    # For position of each face:
    for x, y, w, h in faces:
        # Crop the face
        
        frame = gray[y+10:y+h-10, x+10:x+w-10]
        try:
            roi = cv2.resize(frame, (resolution, resolution))
        except Exception as e:
            frame = gray[y:y+h, x:x+w]
            roi = cv2.resize(frame, (resolution, resolution))
        roi = processing.correct_lighting(roi)
        # roi = processing.contrast(roi)
        cv2.imwrite("vid.jpg", roi)
        
        # Predict the name of the face
        name = model.predict(pca.transform([roi.flatten()]))[0]
        text = str(name)
        color = (255, 255, 0)

        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
    return img