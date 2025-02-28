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

# Configuration
FACE_DB_PATH = "Dataset"
DETECTION_INTERVAL = 0.1  # seconds between detections (faster for continuous recognition)
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence for a match
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

for index in range(4):  # Try camera indices 0, 1, 2, 3
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera index {index} works!")
        cap.release()
        break

    
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
def speak_greeting(name):
    def speak_task():
        greeting = f"Hi, {name}"
        tts_engine.say(greeting)
        tts_engine.runAndWait()
    
    greeting_thread = threading.Thread(target=speak_task)
    greeting_thread.daemon = True
    greeting_thread.start()

# Safe camera initialization function
def initialize_camera():
    global camera
    
    try:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
                time.sleep(1)

            # Try different camera indices
            for cam_index in [0, 1, -1, 2, 3]:  # Extended camera indices
                print(f"Trying camera index {cam_index}")
                camera = cv2.VideoCapture(cam_index)
                
                if camera.isOpened():
                    # Test frame capture
                    ret, _ = camera.read()
                    if ret:
                        print(f"Camera {cam_index} initialized")
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return True
                    else:
                        camera.release()
                        camera = None

            print("All camera indices failed")
            return False
            
    except Exception as e:
        print(f"Camera init error: {e}")
        return False
    

# Function to run face recognition in a separate thread
def face_recognition_thread():
    global recognition_active, camera
    
    print("Starting face recognition thread...")
    
    try:
        # Load known faces when recognition starts
        load_known_faces()
        
        # Initialize camera if needed
        camera_initialized = False
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera_initialized = initialize_camera()
            else:
                camera_initialized = True
                
        if not camera_initialized:
            print("Error: Could not initialize camera")
            recognition_active = False
            return
        
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        
        last_recognition_time = {}
        recognition_cooldown = 5  # seconds between recognizing the same person
        
        consecutive_failures = 0
        max_failures = 5
        
        while recognition_active:
            # Grab a single frame of video
            with camera_lock:
                if camera is None or not camera.isOpened():
                    consecutive_failures += 1
                    print(f"Camera not available (failure {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        print("Too many camera failures, reinitializing...")
                        initialize_camera()
                        consecutive_failures = 0
                        
                    time.sleep(1)
                    continue
                    
                ret, frame = camera.read()
                
                if not ret:
                    consecutive_failures += 1
                    print(f"Failed to grab frame (failure {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        print("Too many frame grab failures, reinitializing camera...")
                        initialize_camera()
                        consecutive_failures = 0
                        
                    time.sleep(0.5)
                    continue
                    
                # Reset failure counter when successfully getting a frame
                consecutive_failures = 0
            
            # Only process every other frame to save CPU
            if process_this_frame:
                # Resize frame to 1/4 size for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert from BGR to RGB (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find all faces in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Check if the face matches any known face
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=CONFIDENCE_THRESHOLD)
                    name = "Unknown"
                    
                    # Use the known face with the smallest distance to the new face
                    if len(known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                    
                    face_names.append(name)
                    
                    # If we recognized someone (not "Unknown"), log attendance and speak greeting
                    if name != "Unknown":
                        current_time = time.time()
                        
                        # Check if we recently recognized this person (to avoid multiple logs)
                        if name not in last_recognition_time or (current_time - last_recognition_time[name]) > recognition_cooldown:
                            last_recognition_time[name] = current_time
                            
                            # Log attendance
                            today = datetime.date.today().isoformat()
                            if today not in attendance_log:
                                attendance_log[today] = set()
                            
                            # Only add to attendance if not already present today
                            if name not in attendance_log[today]:
                                attendance_log[today].add(name)
                                # Update Excel log
                                update_excel_log(name)
                            
                            recognition_event = {
                                             "name": str(name),
                                             "time": datetime.datetime.now().strftime("%H:%M:%S"),
                                             "attendance_count": len(attendance_log[today])
                                                }

                                # Add validation before adding to events
                            if add_recognition_event(recognition_event):
                                  print(f"Recognized: {name} at {recognition_event['time']}")
                            else:
                                  print(f"Failed to add recognition event for: {name}")
                            
                            # Voice greeting with cooldown
                            if name not in voice_greeting_cooldown or (current_time - voice_greeting_cooldown[name]) > GREETING_COOLDOWN_TIME:
                                voice_greeting_cooldown[name] = current_time
                                speak_greeting(name)
            
            process_this_frame = not process_this_frame
            
            # Draw boxes around faces and label them in the frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since we detected on scaled image
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw a box around the face - green for known, red for unknown
                box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                
                # Draw a label with the name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            
            # Store the processed frame for streaming
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                latest_frame_buffer = buffer.tobytes()
                
                # Add the frame to the global frame buffer for streaming
                add_frame_to_buffer(latest_frame_buffer)
            except Exception as e:
                print(f"Error encoding frame: {e}")
            
            # Control the frame rate
            time.sleep(DETECTION_INTERVAL)
    
    except Exception as e:
        print(f"Error in face recognition thread: {e}")
    
    finally:
        print("Face recognition thread stopped")
        recognition_active = False
        
        # Release camera resources
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None

# Function to update Excel attendance log
def update_excel_log(employee_name):
    try:
        # Get current month and year
        now = datetime.datetime.now()
        month_year = now.strftime("%B-%Y")  # e.g., "February-2025"
        excel_file = os.path.join(EXCEL_LOG_PATH, f"{month_year}.xlsx")
        
        today = datetime.date.today().isoformat()
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Load existing data or create new file
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file, engine='openpyxl')
            except Exception as e:
                print(f"Error reading existing Excel file: {e}")
                df = pd.DataFrame(columns=["Date", "Name", "Entry Time"])
        else:
            df = pd.DataFrame(columns=["Date", "Name", "Entry Time"])
        
        # Check if employee already logged for today
        if not ((df["Date"] == today) & (df["Name"] == employee_name)).any():
            # Add new entry
            new_entry = pd.DataFrame({
                "Date": [today],
                "Name": [employee_name],
                "Entry Time": [current_time]
            })
            
            df = pd.concat([df, new_entry], ignore_index=True)
            
            # Save the updated file
            try:
                df.to_excel(excel_file, index=False, engine='openpyxl')
                print(f"Updated Excel log for {employee_name}")
            except Exception as e:
                print(f"Error saving Excel file: {e}")
        
    except Exception as e:
        print(f"Error updating Excel log: {e}")

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
    global recognition_active
    return {
        "success": True,
        "recognition_active": recognition_active,
        "camera_available": camera is not None and (camera.isOpened() if camera is not None else False),
        "known_faces_count": len(known_face_names)
    }

# Modify the recognized_faces_stream function
@app.get("/recognized_faces")
async def recognized_faces_stream():
    async def event_stream():
        connection_id = id(asyncio.current_task())
        print(f"New SSE connection established: {connection_id}")

        try:
            active_sse_connections.add(connection_id)
            last_sent_index = 0

            yield f"data: {json.dumps({'status': 'connected'})}\n\n"

            while True:
                try:
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
                                print(f"Sent event to client {connection_id}: {safe_event['name']}")

                        last_sent_index = len(events)

                    yield ":keep-alive\n\n"
                    await asyncio.sleep(1.0)

                except Exception as e:
                    print(f"Error in SSE event loop: {str(e)}")
                    await asyncio.sleep(2.0)

        except asyncio.CancelledError:
            print(f"SSE connection {connection_id} cancelled")
        except Exception as e:
            print(f"Fatal SSE connection error: {str(e)}")
        finally:
            active_sse_connections.discard(connection_id)
            print(f"SSE connection {connection_id} closed. Active connections: {len(active_sse_connections)}")

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
                    # Add text to black frame
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
    
    return Response(
        content=generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache"
        }
    )

# Modified start_recognition and stop_recognition functions to ensure camera only runs when needed
@app.post("/start_recognition")
async def start_recognition():
    global recognition_active, recognition_thread, camera
    
    if recognition_active:
        return {"success": False, "status": "Recognition is already running"}

    try:
        init_attendance_log()

        # Clear previous recognition data
        with recognition_events_lock:
            global recognition_events
            recognition_events = []

        # Camera initialization with retries
        max_retries = 3
        for attempt in range(max_retries):
            camera_success = initialize_camera()
            if camera_success:
                break
            time.sleep(1)
        else:
            return {"success": False, "status": "Camera initialization failed after 3 attempts"}

        # Start recognition thread if not already running
        recognition_active = True

        if recognition_thread and recognition_thread.is_alive():
            print("Warning: Old recognition thread is still running. Restarting...")
            recognition_active = False
            recognition_thread.join(timeout=2.0)

        recognition_thread = threading.Thread(target=face_recognition_thread)
        recognition_thread.daemon = True
        recognition_thread.start()

        return {"success": True, "status": "Recognition started"}

    except Exception as e:
        recognition_active = False
        with camera_lock:
            if camera:
                camera.release()
                camera = None
        return {"success": False, "status": f"Error: {str(e)}"}


@app.post("/stop_recognition")
async def stop_recognition():
    global recognition_active, recognition_thread, camera
    
    if not recognition_active:
        return {"success": False, "status": "Recognition is not running"}

    try:
        # Stop recognition thread
        recognition_active = False

        # Wait for thread to complete
        if recognition_thread and recognition_thread.is_alive():
            recognition_thread.join(timeout=2.0)
            recognition_thread = None  # Set to None after stopping

        # Release camera
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None

        return {"success": True, "status": "Recognition stopped"}

    except Exception as e:
        return {"success": False, "status": f"Error stopping recognition: {str(e)}"}

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
            # Removing potential prefix like 'data:image/jpeg;base64,'
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            # Decode base64 image
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
                return {"success": False, "status": "Multiple faces detected. Please provide an image with only one face."}
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
            
            # Get timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            
            # Save image file
            image_filename = f"{timestamp}.jpg"
            image_path = os.path.join(user_dir, image_filename)
            cv2.imwrite(image_path, img)
            
            # Count existing images for this user
            image_count = len([f for f in os.listdir(user_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            # Train the face model if we have enough images
            if image_count >= MAX_CAPTURE_IMAGES:
                train_faces(safe_name)
            
            return {
                "success": True,
                "status": f"Face image saved successfully for {name}",
                "image_count": image_count,
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