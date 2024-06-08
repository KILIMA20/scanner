
import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the input image
image = cv2.imread('path/to/image.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display the output image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



//face recognition

import face_recognition

# Load the input image
image = face_recognition.load_image_file('path/to/image.jpg')

# Detect faces in the image
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

# Predict gender and age for each detected face
for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
    # Perform gender prediction (you'll need a trained model for this)
    gender_prediction = predict_gender(encoding)

    # Perform age prediction (you'll need a trained model for this)
    age_prediction = predict_age(encoding)

    # Draw rectangles around the detected faces with predicted labels
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(image, f'Gender: {gender_prediction}', (left + 10, top - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)
    cv2.putText(image, f'Age: {age_prediction}', (left + 10, top - 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.9,(0 ,255 ,0) ,2)

# Display the output image with detected faces and predictions
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
