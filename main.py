from flask import Flask, request, jsonify
import cv2
import dlib
import math
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def MainAccess():
    response_data = {"message": "A7A"}
    return jsonify(response_data)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the dlib shape predictor for facial landmark detection
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    # Read the image data from the request
    frame_bytes = request.data

    # Convert the image data to a NumPy array
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect the face in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, detect the facial landmarks and measure the distances
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Detect the facial landmarks using the dlib shape predictor
        shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))

        # Extract the x and y coordinates of the left and right eyebrows and the nose tip
        left_eyebrow_x = shape.part(20).x
        left_eyebrow_y = shape.part(20).y
        right_eyebrow_x = shape.part(25).x
        right_eyebrow_y = shape.part(25).y
        nose_tip_x = shape.part(34).x
        nose_tip_y = shape.part(34).y

        # Calculate the distances between the left and right eyebrows and the nose tip
        distance_left_eyebrow_nose_tip = math.sqrt((left_eyebrow_x - nose_tip_x) ** 2 + (left_eyebrow_y - nose_tip_y) ** 2)
        distance_right_eyebrow_nose_tip = math.sqrt((right_eyebrow_x - nose_tip_x) ** 2 + (right_eyebrow_y - nose_tip_y) ** 2)

        # Output the result
        cv2.putText(frame, "Distance between left eyebrow and nose tip: {:.2f}".format(distance_left_eyebrow_nose_tip), (x-100, y-150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Distance between right eyebrow and nose tip: {:.2f}".format(distance_right_eyebrow_nose_tip), (x-100, y-130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if distance_left_eyebrow_nose_tip + 5 < distance_right_eyebrow_nose_tip:
            cv2.putText(frame, "WARNING: Left eyebrow closer to nose than right eyebrow!", (x-150, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Convert the processed frame back to bytes
    _, processed_image = cv2.imencode('.jpg', frame)
    processed_image_bytes = processed_image.tobytes()

    # Return the processed image as the response
    return processed_image_bytes

if __name__ == '__main__':
    app.run()
