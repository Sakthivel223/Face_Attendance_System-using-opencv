from getpass import getuser
import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import threading
import time
import json
import os
import numpy as np
import face_recognition
import datetime
import base64
from typing import List, Dict, Set, Optional
from pydantic import BaseModel
import pyttsx3
import pandas as pd
import asyncio
from threading import Lock

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
recognition_active = False
recognition_thread = None
known_faces = {}
known_face_encodings = []
known_face_names = []
attendance_log = {}
active_sse_connections = set()
log_lock = Lock()
GREETING_COOLDOWN = 300  # 5 minutes cooldown per person
recognized_users = {}

# Configuration
FACE_DB_PATH = "Dataset"
DETECTION_INTERVAL = 0.1  # seconds between detections (faster for continuous recognition)
CONFIDENCE_THRESHOLD = 0.8  # minimum confidence for a match
CAMERA_INDEX = 0  # default camera
MAX_CAPTURE_IMAGES = 10  # number of images to capture for registration
EXCEL_LOG_PATH = "AttendanceData"  # Excel logs directory

# Initialize camera
camera = None
camera_lock = threading.Lock()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speaking rate
tts_engine.setProperty('volume', 0.9)  # Volume (0 to 1)

# Voice greeting cooldown to prevent repeated greetings
voice_greeting_cooldown = {}
GREETING_COOLDOWN_TIME = 10  # seconds between greetings for the same person



    
# Ensure necessary directories exist
if not os.path.exists(FACE_DB_PATH):
    os.makedirs(FACE_DB_PATH)
    print(f"Created face database directory: {FACE_DB_PATH}")

if not os.path.exists(EXCEL_LOG_PATH):
    os.makedirs(EXCEL_LOG_PATH)
    print(f"Created Excel logs directory: {EXCEL_LOG_PATH}")

# Load known faces from database
def load_known_faces():
    global known_face_encodings, known_face_names
    
    known_face_encodings = []
    known_face_names = []
    
    print("Loading known faces...")
    
    for person_name in os.listdir(FACE_DB_PATH):
        person_dir = os.path.join(FACE_DB_PATH, person_name)
        if os.path.isdir(person_dir):
            # Look for face encodings saved as JSON
            encoding_path = os.path.join(person_dir, "encoding.json")
            if os.path.exists(encoding_path):
                try:
                    with open(encoding_path, 'r') as f:
                        face_data = json.load(f)
                        if 'encoding' in face_data:
                            face_encoding = np.array(face_data['encoding'])
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(person_name)
                            print(f"Loaded encoding for: {person_name}")
                except Exception as e:
                    print(f"Error loading face encoding from {encoding_path}: {e}")
                continue

            # If no encoding.json exists, check for image files
            for filename in os.listdir(person_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        image_path = os.path.join(person_dir, filename)
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            face_encoding = face_encodings[0]
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(person_name)
                            # Break after first successful encoding per image
                            break
                        else:
                            print(f"No face encodings found in image: {image_path}")
                    except Exception as e:
                        print(f"Error loading face encoding from {image_path}: {e}")
    
    print(f"Loaded {len(known_face_encodings)} face encodings from database")

# Initialize attendance log for today
def init_attendance_log():
    global attendance_log
    today = datetime.date.today().isoformat()
    if today not in attendance_log:
        attendance_log[today] = set()

# Text to speech function with threading to avoid blocking
def speak_greeting(name, is_exit=False):
    if name in voice_greeting_cooldown and (time.time() - voice_greeting_cooldown[name]) < GREETING_COOLDOWN_TIME:
        return  # Skip greeting if cooldown is active

    def speak_task():
        message = f"Hi, {name}" if not is_exit else f"Goodbye, {name}. See you next time!"
        tts_engine.say(message)
        tts_engine.runAndWait()

    greeting_thread = threading.Thread(target=speak_task)
    greeting_thread.daemon = True
    greeting_thread.start()

    voice_greeting_cooldown[name] = time.time()  # ✅ Store the last greeting time

# Safe camera initialization function
def force_stop_camera():
    global camera

    with camera_lock:
        if camera is not None:
            print("🔴 Force stopping existing camera process...")
            camera.release()
            camera = None
            time.sleep(1)  # Allow time for the camera to reset

def initialize_camera():
    global camera
    force_stop_camera()  # ✅ Ensure previous instance is stopped
    with camera_lock:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
    return False

# Function to run face recognition in a separate thread
def face_recognition_thread():
    global recognition_active, camera

    print("✅ Starting face recognition thread...")

    try:
        load_known_faces()

        with camera_lock:
            if camera is None or not camera.isOpened():
                print("🚨 Camera not initialized. Stopping recognition.")
                recognition_active = False
                return

        while recognition_active:
            with camera_lock:
                ret, frame = camera.read()
                if not ret:
                    continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < CONFIDENCE_THRESHOLD:
                      name = known_face_names[best_match_index]
                else:
                      name = "Unknown"

                face_names.append(name)


            # Draw bounding boxes
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            _, buffer = cv2.imencode('.jpg', frame)
            latest_frame_buffer = buffer.tobytes()
            add_frame_to_buffer(latest_frame_buffer)

            time.sleep(0.1)

    except Exception as e:
        print(f"Error in face recognition thread: {e}")
    finally:
        recognition_active = False
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None



def save_to_excel(excel_file, date, name, entry_time, exit_time):
    """ Saves or updates attendance logs in an Excel file. """
    
    if not os.path.exists(EXCEL_LOG_PATH):
        os.makedirs(EXCEL_LOG_PATH)

    # Load existing file or create a new one
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file, engine="openpyxl")
        except Exception as e:
            print(f"Error reading existing Excel file: {e}")
            df = pd.DataFrame(columns=["Date", "Name", "Entry Time", "Exit Time"])
    else:
        df = pd.DataFrame(columns=["Date", "Name", "Entry Time", "Exit Time"])

    # Check if the user already has an entry today
    existing_entry = df[(df["Date"] == date) & (df["Name"] == name)]

    if existing_entry.empty:
        # If no entry exists, add a new row
        new_entry = pd.DataFrame({"Date": [date], "Name": [name], "Entry Time": [entry_time], "Exit Time": [""]})
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        # Update exit time only if it's empty
        df.loc[(df["Date"] == date) & (df["Name"] == name), "Exit Time"] = exit_time

    # Save the updated DataFrame to Excel
    try:
        df.to_excel(excel_file, index=False, engine="openpyxl")
        print(f"✅ Successfully updated {excel_file}")
    except Exception as e:
        print(f"❌ Error saving Excel file: {e}")


# Function to update Excel attendance log
def update_excel_log(name, is_exit=False):
    """ Updates the Excel log with entry/exit times. """
    global attendance_log
    now = datetime.datetime.now()
    today = now.date().isoformat()
    current_time = now.strftime("%H:%M:%S")
    month_year = now.strftime("%B-%Y")
    excel_file = os.path.join(EXCEL_LOG_PATH, f"{month_year}.xlsx")

    with log_lock:
        if today not in attendance_log:
            attendance_log[today] = {}

        if name not in attendance_log[today]:
            attendance_log[today][name] = {"Entry Time": current_time, "Exit Time": ""}
            speak_greeting(name, is_exit=False)

        elif is_exit and attendance_log[today][name]["Exit Time"] == "":
            attendance_log[today][name]["Exit Time"] = current_time
            speak_greeting(name, is_exit=True)

    # ✅ Fixed function call
    log_entry = attendance_log[today][name]
    save_to_excel(excel_file, today, name, log_entry["Entry Time"], log_entry["Exit Time"])



# Function to train faces for a specific user
def train_faces(username: str):
    user_dir = os.path.join(FACE_DB_PATH, username)
    
    if not os.path.exists(user_dir):
        print(f"User directory not found: {user_dir}")
        return False
    
    # Find all image files for this user
    face_encodings = []
    for filename in os.listdir(user_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(user_dir, filename)
                image = face_recognition.load
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    face_encodings.append(encodings[0])
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
    
    if not face_encodings:
        print(f"No valid face encodings found for user: {username}")
        return False
    
    # Average the encodings for better recognition
    average_encoding = np.mean(face_encodings, axis=0)
    
    # Save the encoding as JSON
    encoding_file = os.path.join(user_dir, "encoding.json")
    with open(encoding_file, 'w') as f:
        json.dump({"encoding": average_encoding.tolist()}, f)
    
    print(f"Successfully trained faces for user: {username}")
    
    # Reload known faces if recognition is active
    if recognition_active:
        load_known_faces()
    
    return True

# Frame buffer for streaming video
frame_buffer = []
frame_buffer_lock = threading.Lock()

def add_frame_to_buffer(frame_bytes):
    with frame_buffer_lock:
        global frame_buffer
        frame_buffer.append(frame_bytes)
        
        # Keep only the last 10 frames
        if len(frame_buffer) > 10:
            frame_buffer = frame_buffer[-10:]

def get_latest_frame():
    with frame_buffer_lock:
        if not frame_buffer:
            return None
        return frame_buffer[-1]

# Recognition events for SSE
recognition_events = []
recognition_events_lock = threading.Lock()

def add_recognition_event(event):
    with recognition_events_lock:
        global recognition_events
        try:
            # Validate event format
            if not isinstance(event, dict):
                print(f"Invalid event type: {type(event)}")
                return False
                
            required_keys = ['name', 'time', 'attendance_count']
            missing_keys = [key for key in required_keys if key not in event]
            if missing_keys:
                print(f"Event missing required keys: {missing_keys}")
                return False
            
            # Create a clean copy of the event with validated data types
            safe_event = {
                "name": str(event.get("name", "Unknown")),
                "time": str(event.get("time", "")),
                "attendance_count": int(event.get("attendance_count", 0))
            }
            
            # Add validated event
            recognition_events.append(safe_event)
            
            # Keep only the last 50 events
            if len(recognition_events) > 50:
                recognition_events = recognition_events[-50:]
            
            return True
                
        except Exception as e:
            print(f"Error adding recognition event: {e}")
            return False
        
def get_recognition_events():
    with recognition_events_lock:
        return list(recognition_events)

# Pydantic models for request validation
class NameRequest(BaseModel):
    name: str

class ImageRequest(BaseModel):
    name: str
    image: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.api_route("/update", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def update(request: Request):
    return templates.TemplateResponse("update.html", {"request": request})

@app.get("/ping")
async def ping():
    return {"success": True, "message": "pong"}

@app.get("/check_status")
async def check_status():
    global recognition_active, camera

    with camera_lock:
        camera_status = "not_initialized"
        if camera and camera.isOpened():
            ret, _ = camera.read()
            camera_status = "available" if ret else "not_available"

    return {
        "success": True,
        "recognition_active": recognition_active,
        "camera_available": camera_status,
        "known_faces_count": len(known_face_names)
    }


# Modify the recognized_faces_stream function
@app.get("/recognized_faces")
async def recognized_faces_stream():
    async def event_stream():
        last_sent_index = 0

        yield f"data: {json.dumps({'status': 'connected'})}\n\n"

        while recognition_active:
            events = get_recognition_events()

            if events and len(events) > last_sent_index:
                for i, event in enumerate(events[last_sent_index:], start=last_sent_index):
                    if isinstance(event, dict) and 'name' in event:
                        safe_event = {
                            "name": str(event.get("name", "Unknown")),
                            "time": str(event.get("time", "")),
                            "attendance_count": int(event.get("attendance_count", 0))
                        }

                        yield f"data: {json.dumps(safe_event)}\n\n"
                        last_sent_index = i + 1  # Update last index

            yield ":keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Fixed video_feed function to better handle camera state
@app.get("/video_feed")
async def video_feed():
    async def generate_frames():
        while True:
            try:
                frame = get_latest_frame()
                if frame is not None:
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield frame
                    yield b'\r\n'
                else:
                    # If no frame is available, send a placeholder image
                    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder_frame, "Camera not available", (120, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    _, buffer = cv2.imencode('.jpg', placeholder_frame)
                    frame_bytes = buffer.tobytes()
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield frame_bytes
                    yield b'\r\n'
                
                await asyncio.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Error in video feed: {e}")
                # Send an error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)}", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                _, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                yield b'--frame\r\n'
                yield b'Content-Type: image/jpeg\r\n\r\n'
                yield frame_bytes
                yield b'\r\n'
                
                await asyncio.sleep(1)  # Delay after error

    # ✅ FIX: Use StreamingResponse instead of return inside the generator
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# Modified start_recognition and stop_recognition functions to ensure camera only runs when needed
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@app.post("/start_recognition")
async def start_recognition():
    global recognition_active, recognition_thread, camera

    print(f"🟢 Request to start recognition. Current status: recognition_active={recognition_active}")

    with camera_lock:
        if recognition_active:
            return {"success": False, "status": "Recognition is already running"}

        force_stop_camera()  # Ensure previous camera instance is stopped
        if not initialize_camera():
            return {"success": False, "status": "Failed to initialize camera"}

        recognition_active = True
        recognition_thread = threading.Thread(target=face_recognition_thread, daemon=True)
        recognition_thread.start()

    return {"success": True, "status": "Recognition started successfully"}

@app.post("/stop_recognition")
async def stop_recognition():
    global recognition_active, recognition_thread, camera

    if not recognition_active:
        return {"success": False, "status": "Recognition is not running"}

    recognition_active = False

    if recognition_thread and recognition_thread.is_alive():
        recognition_thread.join(timeout=2.0)
        recognition_thread = None  # Reset thread

    force_stop_camera()  # Ensure camera is released

    return {"success": True, "status": "Recognition stopped"}


@app.get("/get_known_users")
async def get_known_users():
    try:
        # Get all directories in the Dataset folder
        users = []
        if os.path.exists(FACE_DB_PATH):
            for item in os.listdir(FACE_DB_PATH):
                if os.path.isdir(os.path.join(FACE_DB_PATH, item)):
                    users.append(item)
        
        return {"success": True, "users": users}
    
    except Exception as e:
        return {"success": False, "status": f"Error getting users: {str(e)}"}

@app.get("/get_today_attendance")
async def get_today_attendance():
    try:
        # Initialize attendance log if needed
        init_attendance_log()
        
        today = datetime.date.today().isoformat()
        count = len(attendance_log.get(today, set()))
        attended_users = list(attendance_log.get(today, set()))
        
        return {
            "success": True,
            "count": count,
            "users": attended_users,
            "date": today
        }
    
    except Exception as e:
        return {"success": False, "status": f"Error getting attendance: {str(e)}"}

@app.post("/register_face")
async def register_face(data: ImageRequest):
    try:
        name = data.name
        image_data = data.image

        # Validate name
        if not name or len(name.strip()) == 0:
            return {"success": False, "status": "Invalid name"}

        # Clean the name to use as directory name
        safe_name = ''.join(c for c in name if c.isalnum() or c in [' ', '_', '-']).strip()
        safe_name = safe_name.replace(' ', '_')

        if not safe_name:
            return {"success": False, "status": "Name contains no valid characters"}

        # Create user directory if it doesn't exist
        user_dir = os.path.join(FACE_DB_PATH, safe_name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Process image data from base64
        try:
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]

            img_data = base64.b64decode(image_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"success": False, "status": "Could not decode image"}

            # Convert to RGB (face_recognition uses RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            face_locations = face_recognition.face_locations(rgb_img)

            if len(face_locations) == 0:
                return {"success": False, "status": "No face detected in the image"}

            if len(face_locations) > 1:
                return {"success": False, "status": "Multiple faces detected. Please use an image with only one face."}

            # Get face encoding
            face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

            # Save image file
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            image_filename = f"{timestamp}.jpg"
            image_path = os.path.join(user_dir, image_filename)
            cv2.imwrite(image_path, img)

            # Train the face model if encoding.json is missing
            encoding_file = os.path.join(user_dir, "encoding.json")
            if not os.path.exists(encoding_file):
                train_faces(safe_name)  # ✅ Train faces when adding new users

            return {
                "success": True,
                "status": f"Face image saved successfully for {name}",
                "image_count": len(os.listdir(user_dir)),
                "max_images": MAX_CAPTURE_IMAGES
            }

        except Exception as e:
            return {"success": False, "status": f"Error processing image: {str(e)}"}

    except Exception as e:
        return {"success": False, "status": f"Server error: {str(e)}"}


@app.post("/delete_user")
async def delete_user(data: NameRequest):
    try:
        name = data.name
        
        # Validate name
        if not name or len(name.strip()) == 0:
            return {"success": False, "status": "Invalid name"}
        
        # Check if user directory exists
        user_dir = os.path.join(FACE_DB_PATH, name)
        if not os.path.exists(user_dir):
            return {"success": False, "status": f"User '{name}' not found"}
        
        # Delete user directory
        import shutil
        shutil.rmtree(user_dir)
        
        # Reload known faces if recognition is active
        if recognition_active:
            load_known_faces()
        
        return {"success": True, "status": f"User '{name}' deleted successfully"}
        
    except Exception as e:
        return {"success": False, "status": f"Server error: {str(e)}"}

@app.get("/capture_image")
async def capture_image():
    global camera
    
    try:
        # Initialize camera if needed
        camera_initialized = False
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera_initialized = initialize_camera()
            else:
                camera_initialized = True
                
        if not camera_initialized:
            return {"success": False, "status": "Could not initialize camera"}
        
        # Capture frame
        with camera_lock:
            # Take multiple frames to adjust exposure
            for _ in range(5):  # Skip first few frames to allow camera to adjust
                ret, _ = camera.read()
                time.sleep(0.1)
                
            # Capture actual frame for processing
            ret, frame = camera.read()
            
        if not ret:
            return {"success": False, "status": "Failed to capture image"}
        
        # Detect face and draw bounding box
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{jpg_as_text}",
            "face_detected": len(face_locations) > 0
        }
        
    except Exception as e:
        return {"success": False, "status": f"Error capturing image: {str(e)}"}

@app.post("/capture_registration_images")
async def capture_registration_images(data: NameRequest):
    global camera
    
    try:
        name = data.name
        
        # Validate name
        if not name or len(name.strip()) == 0:
            return {"success": False, "status": "Invalid name"}
        
        # Clean the name to use as directory name
        safe_name = ''.join(c for c in name if c.isalnum() or c in [' ', '_', '-']).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            return {"success": False, "status": "Name contains no valid characters"}
        
        # Create user directory if it doesn't exist
        user_dir = os.path.join(FACE_DB_PATH, safe_name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # Initialize camera if needed
        camera_initialized = False
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera_initialized = initialize_camera()
            else:
                camera_initialized = True
                
        if not camera_initialized:
            return {"success": False, "status": "Could not initialize camera"}
        
        captured_images = []
        face_detected_count = 0
        
        # Capture multiple images
        for i in range(MAX_CAPTURE_IMAGES):
            with camera_lock:
                # Wait a moment between captures
                time.sleep(0.5)
                ret, frame = camera.read()
                
            if not ret:
                return {"success": False, "status": "Failed to capture image"}
            
            # Detect face in the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                # Skip this image if no face is detected
                continue
            
            if len(face_locations) > 1:
                # Skip this image if multiple faces are detected
                continue
            
            # Draw a box around the face
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Get timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            
            # Save the image to user directory
            image_filename = f"{timestamp}.jpg"
            image_path = os.path.join(user_dir, image_filename)
            cv2.imwrite(image_path, frame)
            
            # Convert to base64 for response
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            captured_images.append({
                "image": f"data:image/jpeg;base64,{jpg_as_text}",
                "path": image_path
            })
            
            face_detected_count += 1
            
            # Break if we have enough images with faces
            if face_detected_count >= MAX_CAPTURE_IMAGES:
                break
        
        # Train the face model with the captured images
        if face_detected_count > 0:
            train_success = train_faces(safe_name)
            training_status = "Face model trained successfully" if train_success else "Failed to train face model"
        else:
            training_status = "No faces detected in captured images"
        
        return {
            "success": face_detected_count > 0,
            "status": training_status,
            "images_captured": face_detected_count,
            "captured_images": captured_images
        }
        
    except Exception as e:
        return {"success": False, "status": f"Error capturing images: {str(e)}"}

@app.get("/excel_logs")
async def get_excel_logs():
    try:
        logs = []
        if os.path.exists(EXCEL_LOG_PATH):
            for filename in os.listdir(EXCEL_LOG_PATH):
                if filename.endswith('.xlsx'):
                    logs.append(filename)
        
        return {"success": True, "logs": logs}
        
    except Exception as e:
        return {"success": False, "status": f"Error getting logs: {str(e)}"}

@app.get("/download_excel/{filename}")
async def download_excel(filename: str):
    try:
        file_path = os.path.join(EXCEL_LOG_PATH, filename)
        if not os.path.exists(file_path):
            return {"success": False, "status": "File not found"}
        
        return Response(
            content=open(file_path, "rb").read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        return {"success": False, "status": f"Error downloading file: {str(e)}"}
    
@app.get("/sse_debug")
async def sse_debug():
    """Diagnostic endpoint to check SSE system status"""
    try:
        # Get current events for inspection
        events = get_recognition_events()
        
        # Basic validation of stored events
        valid_events = []
        invalid_events = []
        
        for i, event in enumerate(events):
            if isinstance(event, dict) and 'name' in event and 'time' in event and 'attendance_count' in event:
                valid_events.append(event)
            else:
                invalid_events.append({
                    "index": i,
                    "type": type(event).__name__,
                    "content": str(event)[:100]  # Truncate long content
                })
        
        return {
            "success": True,
            "active_connections": len(active_sse_connections),
            "total_events": len(events),
            "valid_events": len(valid_events),
            "invalid_events": invalid_events,
            "sample_events": valid_events[-5:] if valid_events else []
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Main entry point
if __name__ == "__main__":
    try:
        print("Starting Face Recognition Attendance System...")
        # Don't load faces at startup, only when recognition starts
        init_attendance_log()
        uvicorn.run(app, host="localhost", port=8000)
    except KeyboardInterrupt:
        print("Server interrupted, shutting down...")
        # Cleanup
        if recognition_active:
            recognition_active = False
            print("Stopping face recognition...")
        
        # Release camera
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
                print("Camera released")
    except Exception as e:
        print(f"Error starting server: {e}")