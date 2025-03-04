from getpass import getuser
import logging
import random
import traceback
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import threading
import asyncio
import traceback
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
KNOWN_FACES_PATH = "known_faces.json"
FACE_DB_PATH = "Dataset"
DETECTION_INTERVAL = 0.2  # seconds between detections (faster for continuous recognition)
CONFIDENCE_THRESHOLD = 0.8  # minimum confidence for a match
CAMERA_INDEX = 0  # default camera
MAX_CAPTURE_IMAGES = 15  # number of images to capture for registration
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




@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("static/icons/favicon.ico")

# Path to saved face encodings

def load_known_faces():
    """Loads known face encodings from JSON and updates global lists."""
    global known_face_encodings, known_face_names

    known_face_encodings.clear()
    known_face_names.clear()

    try:
        if os.path.exists(KNOWN_FACES_PATH):
            with open(KNOWN_FACES_PATH, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                print("⚠️ Invalid JSON format: Expected a dictionary with names as keys.")
                return

            for name, encoding in data.items():
                if isinstance(encoding, list) and len(encoding) == 128:  # FaceNet encodings have 128 values
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                else:
                    print(f"⚠️ Skipping invalid encoding for {name}")

            print(f"✅ Loaded {len(known_face_encodings)} known faces from JSON.")

        else:
            print("⚠️ known_faces.json not found. Face recognition will not work.")

    except Exception as e:
        print(f"❌ Error loading known faces: {str(e)}")


# Initialize attendance log for today
def init_attendance_log():
    global attendance_log
    today = datetime.date.today().isoformat()
    if today not in attendance_log:
        attendance_log[today] = set()

# Text to speech function with threading to avoid blocking
def speak_greeting(name, is_exit=False):
    """Improved voice greetings with cooldown"""
    GREETING_COOLDOWN_TIME = 10  # ✅ Reduce cooldown to 10 seconds

    message = ""
    if is_exit:
        messages = [
            f"Goodbye {name}!",
            f"See you later {name}!",
            f"Take care {name}!"
        ]
        message = random.choice(messages)
    else:
        messages = [
            f"Welcome back {name}!",
            f"Hello {name}!",
            f"Hi {name}!"
        ]
        message = random.choice(messages)

    def speak_task():
        tts_engine.say(message)
        tts_engine.runAndWait()

    threading.Thread(target=speak_task, daemon=True).start()
    voice_greeting_cooldown[name] = time.time()

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
    """Initialize the camera with multiple index attempts."""
    global camera
    force_stop_camera()

    with camera_lock:
        for index in range(3):  # Try different indexes (0, 1, 2)
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                print(f"✅ Camera initialized at index {index}")
                return True
            camera.release()

        print("❌ Failed to initialize camera. No working index found.")
        return False


   

# Modify the face recognition thread
def face_recognition_thread():
    global recognition_active, camera
    print("✅ Starting face recognition thread...")
    
    try:
        load_known_faces()  # Reload faces every time the thread starts
        
        while recognition_active:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    print("Camera not initialized. Reinitializing...")
                    if not initialize_camera():
                        print("Failed to reinitialize camera. Retrying...")
                        time.sleep(1)
                        continue
                
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
            
            # Validate frame dimensions
            if frame is None or frame.size == 0:
                print("Invalid frame received")
                continue
            
            # Debug: Save a test frame to disk
            # cv2.imwrite("debug_frame.jpg", frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_locations = face_recognition.face_locations(rgb_frame)
            print(f"Detected {len(face_locations)} face(s)")
            
            # Face recognition
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                if not known_face_encodings:
                    print("No known faces to compare with")
                    continue
                
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]
                
                print(f"Best match distance: {min_distance:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")
                
                if min_distance < CONFIDENCE_THRESHOLD:
                    name = known_face_names[best_match_index]
                    print(f"Recognized: {name}")
                    
                    # Update attendance and draw UI
                    update_excel_log(name)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                else:
                    print("No match found")
            
            # Update frame buffer
            _, buffer = cv2.imencode('.jpg', frame)
            add_frame_to_buffer(buffer.tobytes())
            
            time.sleep(DETECTION_INTERVAL)
            
    except Exception as e:
        print(f"CRITICAL ERROR in recognition thread: {str(e)}")
        traceback.print_exc()
    finally:
        recognition_active = False
        print("Recognition thread stopped")
        
def save_to_excel(excel_file, date, name, entry_time, exit_time):
    if not os.path.exists(EXCEL_LOG_PATH):
        os.makedirs(EXCEL_LOG_PATH)

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file, engine="openpyxl")
    else:
        df = pd.DataFrame(columns=["Date", "Name", "Entry Time", "Exit Time"])

    existing_entry = df[(df["Date"] == date) & (df["Name"] == name)]

    if existing_entry.empty:
        df = pd.concat([df, pd.DataFrame({"Date": [date], "Name": [name], "Entry Time": [entry_time], "Exit Time": [""]})], ignore_index=True)
    else:
        df.loc[(df["Date"] == date) & (df["Name"] == name), "Exit Time"] = exit_time

    df.to_excel(excel_file, index=False, engine="openpyxl")
    print(f"✅ Excel updated: {excel_file}")

# Function to update Excel attendance log
def update_excel_log(name, is_exit=False):
    global attendance_log
    now = datetime.datetime.now()
    today = now.date().isoformat()
    current_time = now.strftime("%H:%M:%S")

    with log_lock:
        if today not in attendance_log:
            attendance_log[today] = {}

        if name not in attendance_log[today]:
            # First Entry
            attendance_log[today][name] = {
                "entry_time": current_time,
                "exit_time": "",
                "last_seen": now
            }
            speak_greeting(name, is_exit=False)  # ✅ Greet on Entry

        else:
            # Check if last seen was more than 5 minutes ago
            if is_exit and (now - attendance_log[today][name]["last_seen"]).seconds > 300:
                attendance_log[today][name]["exit_time"] = current_time
                speak_greeting(name, is_exit=True)  # ✅ Greet on Exit

            # Update last seen time
            attendance_log[today][name]["last_seen"] = now

    # ✅ Save to Excel
    excel_file = os.path.join(EXCEL_LOG_PATH, f"{datetime.datetime.now().strftime('%B-%Y')}.xlsx")
    save_to_excel(excel_file, today, name, 
                 attendance_log[today][name]["entry_time"],
                 attendance_log[today][name]["exit_time"])


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


# WITH THIS UPDATED VERSION
def add_frame_to_buffer(frame_bytes):
    with frame_buffer_lock:
        global frame_buffer
        frame_buffer = [frame_bytes]
        print(f"Buffer updated with frame size: {len(frame_bytes)} bytes")  # Debug

def get_latest_frame():
    with frame_buffer_lock:
        if not frame_buffer:
            return None
        return frame_buffer[-1]  # This will now return the only frame

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

# Update check_status endpoint
@app.get("/check_status")
async def check_status():
    global recognition_active, camera
    camera_status = "not_initialized"
    with camera_lock:
        if camera and camera.isOpened():
            camera_status = "available"
    return {
        "success": True,
        "recognition_active": recognition_active,
        "camera_available": camera_status,
        "known_faces_count": len(known_face_names),
        "last_seen": datetime.datetime.now().isoformat()  # Add this line
    }

# Modify the recognized_faces_stream function
@app.get("/recognized_faces")
async def recognized_faces_stream():
    async def event_stream():
        last_sent_index = 0
        yield f"data: {json.dumps({'status': 'connected'})}\n\n"
        
        while True:
            try:
                events = get_recognition_events()
                
                if events and len(events) > last_sent_index:
                    for i, event in enumerate(events[last_sent_index:], start=last_sent_index):
                        safe_event = {
                            "name": str(event.get("name", "Unknown")),
                            "time": str(event.get("time", "")),
                            "attendance_count": int(event.get("attendance_count", 0))
                        }
                        yield f"data: {json.dumps(safe_event)}\n\n"
                        last_sent_index = i + 1
                
                # ✅ Send a keep-alive event to prevent disconnects
                yield "data: {}\n\n"
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"SSE Error: {e}")
                await asyncio.sleep(3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Fixed video_feed function to better handle camera state
@app.get("/video_feed")
async def video_feed():
    async def generate_frames():
        while recognition_active:
            frame = get_latest_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.033)  # ~30 FPS
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


# Modified start_recognition and stop_recognition functions to ensure camera only runs when needed
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")



@app.post("/start_recognition")
async def start_recognition():
    global recognition_active

    if recognition_active:
        return JSONResponse(status_code=400, content={"success": False, "status": "Recognition already running"})

    # Initialize the camera
    if not initialize_camera():
        return JSONResponse(status_code=500, content={"success": False, "status": "Failed to initialize camera"})

    # Start recognition thread
    recognition_active = True
    asyncio.create_task(process_face_recognition())

    return JSONResponse(status_code=200, content={"success": True, "status": "Face recognition started"})



def face_recognition_watchdog():
    """Restarts recognition thread on failure"""
    while recognition_active:
        try:
            print("🔄 Starting face recognition thread...")
            face_recognition_thread()
        except Exception as e:
            print(f"❌ Recognition thread crashed: {str(e)}")
            traceback.print_exc()
            time.sleep(1)  # Prevent tight crash loop

async def process_face_recognition():
    """Runs face recognition using known_faces.json for speed."""
    global recognition_active, camera

    print("🔄 Running face recognition (Using JSON Data)")

    try:
        if not known_face_encodings or not known_face_names:
            print("⚠️ No known faces loaded. Attempting to load from JSON.")
            load_known_faces()

        while recognition_active:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    print("⚠️ Camera is not initialized. Attempting to restart.")
                    if not initialize_camera():
                        print("❌ Camera initialization failed. Stopping recognition.")
                        break

                ret, frame = camera.read()
                if not ret:
                    print("⚠️ Failed to capture frame")
                    await asyncio.sleep(0.1)
                    continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Change to "cnn" for better accuracy
            print(f"🔍 Detected {len(face_locations)} face(s)")

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                if not known_face_encodings:
                    print("⚠️ No known faces in database")
                    continue

                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index] if len(distances) > 0 else 1.0

                if min_distance < CONFIDENCE_THRESHOLD:
                    name = known_face_names[best_match_index]
                    print(f"✅ Recognized: {name} (confidence: {1 - min_distance:.2%})")
                    update_excel_log(name)
                    color = (0, 255, 0)  # Green for recognized faces
                else:
                    name = "Unknown"
                    print(f"⚠️ Unknown face detected (confidence: {1 - min_distance:.2%})")
                    color = (0, 0, 255)  # Red for unknown faces

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                update_excel_log(name)
                add_recognition_event({
                    "name": name,
                    "time": datetime.datetime.now().isoformat(),
                    "attendance_count": len(attendance_log.get(datetime.date.today().isoformat(), set()))
                })

            _, buffer = cv2.imencode('.jpg', frame)
            add_frame_to_buffer(buffer.tobytes())

            if not recognition_active:  # Ensure loop stops when requested
                break

            await asyncio.sleep(DETECTION_INTERVAL)

    except Exception as e:
        print(f"❌ Critical error in face recognition: {str(e)}")
        traceback.print_exc()
    finally:
        recognition_active = False
        print("🛑 Face recognition stopped")

@app.post("/stop_recognition")
async def stop_recognition():
    global recognition_active, camera

    if not recognition_active:
        return {"success": False, "status": "Recognition is not running"}

    recognition_active = False  # ✅ Stop face recognition immediately
    await asyncio.sleep(0.5)  # ✅ Allow time for recognition loop to exit
    force_stop_camera()  # ✅ Release camera

    return {"success": True, "status": "Face recognition stopped"}

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