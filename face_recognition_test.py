import cv2
import face_recognition
import numpy as np
import json

# Load known faces from JSON file
def load_known_faces(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    known_face_encodings = [np.array(face['encoding']) for face in data]
    known_face_names = [face['name'] for face in data]
    return known_face_encodings, known_face_names

# Load trained faces
known_face_encodings, known_face_names = load_known_faces("known_faces.json")
print(f"✅ Loaded {len(known_face_encodings)} known faces.")

# Start the webcam
video_capture = cv2.VideoCapture(0)  # 0 for default webcam
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    # Resize frame for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255)  # Default: Red for unknown faces

        # Find best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
            color = (0, 255, 0)  # Green for known faces

        # Scale bounding box back to original size
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Show live video window
    cv2.imshow("Face Recognition Live Stream", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
