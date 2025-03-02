import os
import numpy as np
import face_recognition
import json
from PIL import Image
import time

# Configuration
DATASET_FOLDER = "Dataset"
KNOWN_FACES_FILE = "known_faces.json"
VALID_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MIN_IMAGES_PER_USER = 5  # Minimum images required for training
FACE_DETECTION_MODEL = "hog"  # or "cnn" for better accuracy (slower)

def train_all_faces():
    """Train face recognition model using all users in the dataset"""
    start_time = time.time()
    known_faces = {}
    trained_users = []
    failed_users = []
    
    print(f"🚀 Starting face training process for {DATASET_FOLDER}")
    
    if not os.path.exists(DATASET_FOLDER):
        raise FileNotFoundError(f"Dataset folder '{DATASET_FOLDER}' not found")

    # Get all user directories
    users = [d for d in os.listdir(DATASET_FOLDER) 
             if os.path.isdir(os.path.join(DATASET_FOLDER, d))]
    
    if not users:
        raise ValueError("No user folders found in dataset directory")

    print(f"🔍 Found {len(users)} users in dataset")
    
    for user in users:
        try:
            print(f"\n⚙️ Processing user: {user}")
            user_dir = os.path.join(DATASET_FOLDER, user)
            encodings = []
            valid_images = 0
            
            # Process all images in user directory
            for filename in os.listdir(user_dir):
                if not filename.lower().endswith(tuple(VALID_EXTENSIONS)):
                    continue
                
                image_path = os.path.join(user_dir, filename)
                
                try:
                    # Verify image integrity
                    with Image.open(image_path) as img:
                        img.verify()
                    
                    # Load and process image
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
                    
                    if not face_locations:
                        print(f"  ⚠️ No faces found in {filename}")
                        continue
                        
                    # Get encodings for all faces found in image
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    encodings.extend([e.tolist() for e in face_encodings])
                    valid_images += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {str(e)}")
                    continue
            
            # Quality control checks
            if valid_images < MIN_IMAGES_PER_USER:
                raise ValueError(f"Only {valid_images} valid images found (minimum {MIN_IMAGES_PER_USER} required)")
                
            if not encodings:
                raise ValueError("No valid face encodings generated")
            
            # Create average encoding
            avg_encoding = np.mean(encodings, axis=0).tolist()
            known_faces[user] = avg_encoding
            trained_users.append(user)
            print(f"  ✅ Successfully trained with {valid_images} images")
            
        except Exception as e:
            print(f"  🔥 Failed to train {user}: {str(e)}")
            failed_users.append(user)
    
    # Save results
    with open(KNOWN_FACES_FILE, 'w') as f:
        json.dump(known_faces, f, indent=4)
    
    # Print summary
    print("\n📊 Training Summary:")
    print(f"✅ Successfully trained: {len(trained_users)} users")
    print(f"❌ Failed to train: {len(failed_users)} users")
    print(f"⏱️ Total time: {time.time()-start_time:.2f} seconds")
    
    if trained_users:
        print("\nTrained users:")
        for user in trained_users:
            print(f" - {user}")
    
    if failed_users:
        print("\nFailed users:")
        for user in failed_users:
            print(f" - {user}")
    
    return len(trained_users)

if __name__ == "__main__":
    try:
        trained_count = train_all_faces()
        if trained_count > 0:
            print(f"\n🎉 Training complete! Encodings saved to {KNOWN_FACES_FILE}")
        else:
            print("\n😞 No users were successfully trained")
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")