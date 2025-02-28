import os
import numpy as np
import face_recognition
import json

KNOWN_FACES_FILE = "known_faces.json"

DATASET_FOLDER = "Dataset"

def train_faces(username):
    """Train faces from images in the specified user's folder and store encodings locally."""
    person_folder = os.path.join(DATASET_FOLDER, username)


    known_faces = {}
    
    if not os.path.exists(DATASET_FOLDER):
        print(f"Error: Folder '{DATASET_FOLDER}' not found.")
        return
    
    if not os.path.isdir(person_folder):
        print(f"Error: Folder '{username}' not found.")
        return

    encodings = []
    for filename in os.listdir(person_folder):

        
        image_path = os.path.join(person_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            encodings.append(face_encodings[0].tolist())
    
    if encodings:
        known_faces[username] = np.mean(encodings, axis=0).tolist()
        with open(KNOWN_FACES_FILE, 'w') as f:
            json.dump(known_faces, f, indent=4)
        print(f"Face encodings successfully saved to {KNOWN_FACES_FILE}")
    except Exception as e:
        print(f"Error saving face encodings: {e}")


if __name__ == "__main__":
    train_faces()
