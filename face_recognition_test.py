import time
import cv2
import face_recognition
import numpy as np
import os

def test_face_recognition():
    # Test 1: Verify known face loading
    known_faces_dir = "Dataset/Vasanth Kumar"  # Replace with your test user
    print(f"\n=== Testing face recognition for: {known_faces_dir} ===")
    
    # Load test image
    test_image_path = os.path.join(known_faces_dir, "54.jpg")
    known_image = face_recognition.load_image_file(test_image_path)
    
    # Test 2: Face detection
    face_locations = face_recognition.face_locations(known_image)
    print(f"Face detection test: Found {len(face_locations)} face(s)")
    
    # Test 3: Face encoding
    known_encoding = face_recognition.face_encodings(known_image)[0]
    print("Face encoding test: Successfully generated face encoding")
    
    # Test 4: Live camera comparison
    cap = cv2.VideoCapture(0)
    print("\n=== Live camera test ===")
    print("Look at the camera for 10 seconds...")
    
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in live frame
        live_face_locations = face_recognition.face_locations(rgb_frame)
        live_face_encodings = face_recognition.face_encodings(rgb_frame, live_face_locations)
        
        for face_encoding in live_face_encodings:
            # Compare with known encoding
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            distance = face_recognition.face_distance([known_encoding], face_encoding)
            print(f"Match: {matches[0]} | Confidence: {1 - distance[0]:.2%}")
            
        cv2.imshow('Test Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Test complete ===")

if __name__ == "__main__":
    test_face_recognition()