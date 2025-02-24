from flask import Flask, jsonify, render_template, Response, request, redirect, send_from_directory, url_for
import cv2
import os
import time
import threading
import numpy as np
import json
from datetime import datetime
import face_recognition
from train_face import train_faces

app = Flask(__name__)

# Global variables
camera = None
recognition_active = False
update_mode = False
capture_count = 0
user_name = ""
dataset_path = "Dataset"
known_faces_file = "known_faces.json"
known_face_encodings = []
known_face_names = []
attendance_log = []
camera_lock = threading.Lock()  # Add a lock for thread safety

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='face_recognition.log'
)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def start_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not camera.isOpened():
                logging.error("Failed to open camera")
                return False
            logging.info("Camera started successfully")
            return True
        return True

def stop_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            logging.info("Camera stopped")

def load_known_faces():
    global known_face_encodings, known_face_names
    try:
        if os.path.exists(known_faces_file):
            with open(known_faces_file, "r") as f:
                known_faces = json.load(f)
            known_face_encodings = [np.array(enc) for enc in known_faces.values()]
            known_face_names = list(known_faces.keys())
            logging.info(f"Successfully loaded {len(known_face_names)} known faces")
        else:
            logging.warning(f"Known faces file '{known_faces_file}' not found")
    except Exception as e:
        logging.error(f"Error loading known faces: {e}")
        known_face_encodings = []
        known_face_names = []

def log_attendance(name):
    """Log attendance with timestamp"""
    global attendance_log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"name": name, "timestamp": timestamp}
    
    # Check if this person was already logged in the last minute
    current_time = datetime.now()
    for entry in attendance_log:
        if entry["name"] == name:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            time_diff = (current_time - entry_time).total_seconds()
            if time_diff < 60:  # Less than a minute ago
                return  # Skip logging
    
    attendance_log.append(log_entry)
    
    # Save to file
    with open("attendance_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    logging.info(f"Attendance logged: {name} at {timestamp}")

def generate_frames():
    global recognition_active, update_mode, capture_count, user_name, camera
    
    # Default image when camera is not active
    placeholder_img = cv2.imread('static/icons/placeholder.jpg') if os.path.exists('static/icons/placeholder.jpg') else None
    if placeholder_img is None:
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Camera Inactive", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while True:
        # Check if recognition is active
        if not recognition_active and not update_mode:
            # If camera is running but shouldn't be, stop it
            with camera_lock:
                if camera is not None:
                    stop_camera()
                    
            # Return placeholder image
            _, buffer = cv2.imencode('.jpg', placeholder_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            continue
        
        # Camera should be active now
        with camera_lock:
            if camera is None:
                if not start_camera():
                    # Failed to start camera, show placeholder and try again later
                    _, buffer = cv2.imencode('.jpg', placeholder_img)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.5)
                    continue
            
            success, frame = camera.read()
            
        if not success:
            logging.warning("Failed to read frame from camera")
            # Try to restart the camera if reading fails
            with camera_lock:
                stop_camera()
                
            time.sleep(0.1)
            continue
        
        # Process frame based on mode
        if update_mode:
            # Handle user update mode
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Add text showing capture progress
            cv2.putText(frame, f"Capturing images for: {user_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Images: {capture_count}/10", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                if capture_count < 10:
                    # Ensure directory exists
                    save_path = os.path.join(dataset_path, user_name)
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save cropped face image
                    face_img = frame[y:y+h, x:x+w]
                    img_path = os.path.join(save_path, f"{capture_count}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    capture_count += 1
                    logging.info(f"Captured image {capture_count}/10 for user {user_name}")
                    time.sleep(0.5)  # Delay between captures
            
            # Check if we've completed capturing
            if capture_count >= 10:
                update_mode = False
                logging.info(f"Completed capturing images for {user_name}. Starting training...")
                
                # Start training in a separate thread
                threading.Thread(target=train_faces).start()
                
                # Add completion message
                cv2.putText(frame, "Image capture complete! Training model...", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        elif recognition_active:
            # Load known faces if not already loaded
            if not known_face_names:
                load_known_faces()
                
            # Handle face recognition
            # Convert to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Default as unknown
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    
                    # Compare with known faces if we have any
                    if known_face_encodings:
                        # Calculate face distances
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            
                            # Only consider it a match if the distance is below threshold
                            if face_distances[best_match_index] < 0.6:  # Adjust threshold as needed
                                name = known_face_names[best_match_index]
                                color = (0, 255, 0)  # Green for known faces
                                
                                # Log attendance for known faces
                                log_attendance(name)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw filled rectangle below for name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    # Put name text
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global recognition_active
    # Ensure camera is off when page loads initially
    recognition_active = False
    stop_camera()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global recognition_active
    recognition_active = True
    success = start_camera()
    
    if success:
        # Load known faces on start
        load_known_faces()
        return jsonify({"status": "Recognition started", "success": True}), 200
    else:
        recognition_active = False  # Reset flag if camera failed
        return jsonify({"status": "Failed to start camera", "success": False}), 500

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global recognition_active
    recognition_active = False
    stop_camera()
    return jsonify({"status": "Recognition stopped", "success": True}), 200

@app.route('/update', methods=['GET'])
def update_page():
    global recognition_active, update_mode
    # Ensure camera is off when navigating to update page
    recognition_active = False
    update_mode = False
    stop_camera()
    return render_template('update.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global update_mode, capture_count, user_name, recognition_active
    
    data = request.get_json()
    user_name = data.get('username', '')
    
    if not user_name:
        return jsonify({"status": "Error: Username is required", "success": False}), 400
    
    # Make sure recognition is off when in capture mode
    recognition_active = False
    capture_count = 0
    update_mode = True
    success = start_camera()
    
    if success:
        return jsonify({"status": f"Capturing started for user: {user_name}", "success": True}), 200
    else:
        update_mode = False  # Reset flag if camera failed
        return jsonify({"status": "Failed to start camera", "success": False}), 500

@app.route('/logs')
def view_logs():
    global recognition_active, update_mode
    # Ensure camera is off when viewing logs
    recognition_active = False
    update_mode = False
    stop_camera()
    
    try:
        logs = []
        if os.path.exists("attendance_log.json"):
            with open("attendance_log.json", "r") as f:
                for line in f:
                    logs.append(json.loads(line))
        
        # Sort logs by timestamp (newest first)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return render_template('logs.html', logs=logs)
    except Exception as e:
        logging.error(f"Error loading logs: {e}")
        return render_template('logs.html', logs=[], error=str(e))

@app.route('/get_known_users', methods=['GET'])
def get_known_users():
    try:
        if os.path.exists(known_faces_file):
            with open(known_faces_file, "r") as f:
                known_faces = json.load(f)
            return jsonify({"users": list(known_faces.keys()), "success": True}), 200
        else:
            return jsonify({"users": [], "success": True}), 200
    except Exception as e:
        logging.error(f"Error getting known users: {e}")
        return jsonify({"users": [], "error": str(e), "success": False}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.errorhandler(404)
def page_not_found(e):
    global recognition_active, update_mode
    # Ensure camera is off when navigating to error page
    recognition_active = False
    update_mode = False
    stop_camera()
    return render_template('404.html'), 404

# Add a route for client-side checking of recognition status
@app.route('/check_status', methods=['GET'])
def check_status():
    global recognition_active, update_mode
    return jsonify({
        "recognition_active": recognition_active,
        "update_mode": update_mode,
        "success": True
    }), 200

# Add template context processor to pass recognition status to all templates
@app.context_processor
def inject_recognition_status():
    global recognition_active, update_mode
    return {
        "recognition_active": recognition_active,
        "update_mode": update_mode
    }

# Handle application shutdown
@app.teardown_appcontext
def shutdown_session(exception=None):
    global camera
    # Make sure camera is released when app shuts down
    stop_camera()

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs('static/icons', exist_ok=True)
    
    # Create placeholder image if it doesn't exist
    placeholder_path = 'static/icons/placeholder.jpg'
    if not os.path.exists(placeholder_path):
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Camera Inactive", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(placeholder_path, placeholder_img)
    
    # Register a function to run when server shuts down
    def cleanup_on_exit():
        global camera
        if camera is not None:
            stop_camera()
            logging.info("Camera released during shutdown")
    
    # Register the cleanup function
    import atexit
    atexit.register(cleanup_on_exit)
    
    # Start the app
    app.run(debug=True, host='0.0.0.0', port=5000)