import fractions
import logging
import random
import traceback
from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocketState
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack
import av
import uvicorn
import cv2
import threading
import asyncio
import time
import json
import os
import numpy as np
import face_recognition
import datetime
import base64
from typing import Any, List, Dict, Set, Optional
from pydantic import BaseModel
import pyttsx3
import pandas as pd
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
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence for a match (lowered for better recognition)
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
                if isinstance(encoding, list) and len(encoding) == 128:
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
        attendance_log[today] = {}

# Text to speech function with threading to avoid blocking
def speak_greeting(name, is_exit=False):
    """Improved voice greetings with cooldown"""
    current_time = time.time()
    
    # Check if we should skip this greeting due to cooldown
    if name in voice_greeting_cooldown:
        last_greeting = voice_greeting_cooldown[name]
        if current_time - last_greeting < GREETING_COOLDOWN_TIME:
            print(f"Skipping greeting for {name} (cooldown active)")
            return
    
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
        try:
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in speech: {str(e)}")

    threading.Thread(target=speak_task, daemon=True).start()
    voice_greeting_cooldown[name] = current_time
    print(f"Greeting: {message}")

def force_stop_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("Camera released")
    time.sleep(0.5)  # Allow reset time

# Camera setup
def initialize_camera():
    global camera
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Check if camera opened successfully
            if not camera.isOpened():
                print("Error: Could not open camera.")
                camera = None
                return False
                
            # Read a test frame
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Error: Could not read from camera.")
                camera.release()
                camera = None
                return False
                
            print("Camera initialized successfully")
            return True
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        if camera is not None:
            camera.release()
            camera = None
        return False
    return False

camera = None

@app.post("/webrtc/ice_candidate")
async def ice_candidate(request: Request):
    try:
        data = await request.json()
        candidate = data.get("candidate")
        sdp_mid = data.get("sdpMid")
        sdp_m_line_index = data.get("sdpMLineIndex")
        connection_id = data.get("connectionId")
        
        # Find the appropriate peer connection
        target_pc = None
        for pc in peer_connections:
            if id(pc) == connection_id:
                target_pc = pc
                break
        
        if target_pc and candidate:
            candidate_obj = RTCIceCandidate(
                component=candidate.get("component", 0),
                foundation=candidate.get("foundation", ""),
                ip=candidate.get("ip", ""),
                port=candidate.get("port", 0),
                priority=candidate.get("priority", 0),
                protocol=candidate.get("protocol", ""),
                type=candidate.get("type", ""),
                sdpMid=sdp_mid,
                sdpMLineIndex=sdp_m_line_index
            )
            await target_pc.addIceCandidate(candidate_obj)
            return {"success": True}
        else:
            return {"success": False, "error": "No matching connection found or invalid candidate"}
    except Exception as e:
        print(f"Error processing ICE candidate: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)    
    
# Store active peer connections
peer_connections = set()

class CustomVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns camera frames for WebRTC streaming.
    """
    def __init__(self):
        super().__init__()  # don't forget this!
        self.counter = 0
        self.frames = 0
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
            if not self.cap.isOpened():
                raise Exception("Could not open video device")
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("Camera initialized for WebRTC streaming")
        except Exception as e:
            print(f"Error initializing camera for WebRTC: {str(e)}")
    
    async def recv(self):
        self.frames += 1
        
        # If we have a camera, use it
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # If camera read fails, create a blank frame
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                # Add text showing camera error
                cv2.putText(img, "Camera Error", (220, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Convert from BGR to RGB
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Create a blank frame with message if no camera
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No Camera Available", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to video frame
        video_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        video_frame.pts = self.counter
        video_frame.time_base = fractions.Fraction(1, 30)  # 30fps
        
        self.counter += 1
        return video_frame    
    
    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("WebRTC camera released")

@app.post("/webrtc/offer")
async def offer(request: Request):
    pc = None
    try:
        # Parse the request body
        request_data = await request.json()
        
        # Create a new RTCPeerConnection
        pc = RTCPeerConnection()
        peer_connections.add(pc)
        
        # Add cleanup callback when connection closes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                if pc in peer_connections:
                    peer_connections.discard(pc)
                    print(f"WebRTC connection closed, remaining connections: {len(peer_connections)}")
        
        # Add video track BEFORE processing the offer
        video_track = CustomVideoStreamTrack()
        pc.addTrack(video_track)
        
        # Process the offer
        offer = RTCSessionDescription(sdp=request_data["sdp"], type=request_data["type"])
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Wait for ICE gathering to complete
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
        
        # Return the answer with complete ICE candidates
        return JSONResponse(content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
        
    except Exception as e:
        print(f"Error in WebRTC setup: {str(e)}")
        traceback.print_exc()
        # Clean up if there was an error
        if pc and pc in peer_connections:
            peer_connections.discard(pc)
            await pc.close()
        return JSONResponse(content={"error": str(e)}, status_code=500)
            
@app.get("/video_feed")
async def video_feed():
    """Fallback HTTP video stream if WebRTC fails"""
    async def generate():
        while recognition_active:
            # Get the latest frame
            frame_data = get_latest_frame()
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                # If no frame is available, yield an empty frame
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty_frame, "No Camera Feed", (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', empty_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

def setup_logging():
    """Set up proper logging for the application."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"app_{datetime.datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log startup information
    logging.info("=" * 50)
    logging.info("Face Recognition Attendance System starting up")
    logging.info(f"OpenCV Version: {cv2.__version__}")
    logging.info(f"Camera Index: {CAMERA_INDEX}")
    logging.info(f"Known Faces Path: {KNOWN_FACES_PATH}")
    logging.info(f"Face DB Path: {FACE_DB_PATH}")
    logging.info(f"Excel Log Path: {EXCEL_LOG_PATH}")
    logging.info("=" * 50)

# Use this in your code
def log_error(message, exception=None):
    error_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_msg = f"[{error_time}] ERROR: {message}"
    if exception:
        error_msg += f"\n{traceback.format_exc()}"
    print(error_msg)
    # Log to file
    with open("error_log.txt", "a") as f:
        f.write(error_msg + "\n")

# Frame buffer for streaming video
frame_buffer = []
frame_buffer_lock = threading.Lock()


# Add before face recognition to improve accuracy
def align_face(image, face_location):
    try:
        # Get facial landmarks
        landmarks = face_recognition.face_landmarks(image, [face_location])[0]
        
        # Use eye landmarks for alignment
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return aligned_image
    except Exception:
        # Return original image if alignment fails
        return image
    
# Frame buffer for streaming video
frame_buffer = None  # Store just the latest frame
frame_buffer_lock = threading.Lock()

def update_frame_buffer(frame_bytes):
    """Updates the global frame buffer with the latest frame bytes"""
    with frame_buffer_lock:
        global frame_buffer
        frame_buffer = frame_bytes

def get_latest_frame():
    """Returns the latest frame from the buffer"""
    with frame_buffer_lock:
        return frame_buffer if frame_buffer else None


async def cleanup_video_connections():
    """Close all WebRTC connections properly"""
    global peer_connections
    
    cleanup_tasks = []
    for pc in list(peer_connections):
        try:
            cleanup_tasks.append(pc.close())
        except Exception as e:
            print(f"Error closing peer connection: {e}")
    
    if cleanup_tasks:
        # Wait for all connections to close
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    peer_connections.clear()
    print(f"Closed {len(cleanup_tasks)} WebRTC connections")

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

def save_to_excel(excel_file, date, name, entry_time, exit_time):
    if not os.path.exists(EXCEL_LOG_PATH):
        os.makedirs(EXCEL_LOG_PATH)

    for attempt in range(5):  # Retry up to 5 times
        try:
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file, engine="openpyxl")
            else:
                df = pd.DataFrame(columns=["Date", "Name", "Entry Time", "Exit Time"])

            df["Exit Time"] = df["Exit Time"].astype(str)  
            
            existing_entry = df[(df["Date"] == date) & (df["Name"] == name)]
            
            if existing_entry.empty:
                new_row = pd.DataFrame({"Date": [date], "Name": [name], "Entry Time": [entry_time], "Exit Time": [""]})
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df.loc[(df["Date"] == date) & (df["Name"] == name), "Exit Time"] = str(exit_time)

            with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, index=False)

            print(f"✅ Excel updated: {excel_file}")
            return True

        except PermissionError:
            print(f"⚠️ File locked. Retrying... (Attempt {attempt + 1})")
            time.sleep(1)

    print("❌ Unable to update Excel after multiple attempts.")
    return False

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
            speak_greeting(name, is_exit=False)
            print(f"✅ New entry for {name} at {current_time}")
        else:
            # Check if last seen was more than 5 minutes ago
            time_diff = (now - attendance_log[today][name]["last_seen"]).total_seconds()
            if time_diff > 300:  # 5 minutes - this might be an exit and return
                if attendance_log[today][name]["exit_time"]:
                    # They've returned after an exit
                    speak_greeting(name, is_exit=False)
                    print(f"✅ Return entry for {name} at {current_time}")
                else:
                    # First exit
                    attendance_log[today][name]["exit_time"] = current_time
                    speak_greeting(name, is_exit=True)
                    print(f"✅ Exit recorded for {name} at {current_time}")

            # Update last seen time
            attendance_log[today][name]["last_seen"] = now

    # Save to Excel
    excel_file = os.path.join(EXCEL_LOG_PATH, f"{datetime.datetime.now().strftime('%B-%Y')}.xlsx")
    save_to_excel(excel_file, today, name, 
                 attendance_log[today][name]["entry_time"],
                 attendance_log[today][name]["exit_time"])
    
    # Add recognition event for frontend
    add_recognition_event({
        "name": name,
        "time": now.isoformat(),
        "attendance_count": len(attendance_log.get(today, {}))
    })

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
        "last_seen": datetime.datetime.now().isoformat()
    }

# Modify the recognized_faces_stream function
# Modify the recognized_faces_stream function
@app.get("/recognized_faces")
async def recognized_faces_stream():
    async def event_stream():
        # Add this connection to active connections
        connection_id = id(asyncio.current_task())
        active_sse_connections.add(connection_id)
        logging.info(f"SSE connection established: {connection_id}")
        
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'status': 'connected', 'connection_id': connection_id})}\n\n"
            
            last_sent_index = 0
            keep_alive_counter = 0
            
            while True:
                try:
                    events = get_recognition_events()
                    
                    if events and len(events) > last_sent_index:
                        # Send new events
                        for i in range(last_sent_index, len(events)):
                            event = events[i]
                            yield f"data: {json.dumps(event)}\n\n"
                        last_sent_index = len(events)
                        keep_alive_counter = 0
                    else:
                        # Send keep-alive every 5 seconds instead of 10
                        keep_alive_counter += 1
                        if keep_alive_counter >= 5:  # Reduced from 10 to 5
                            yield f"data: {json.dumps({'status': 'keep-alive', 'timestamp': time.time()})}\n\n"
                            keep_alive_counter = 0
                    
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"SSE Error: {e}")
                    await asyncio.sleep(3)
        finally:
            # Remove this connection when it ends
            active_sse_connections.discard(connection_id)
            logging.info(f"SSE connection closed: {connection_id}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

               
@app.post("/start_recognition")
async def start_recognition():
    global recognition_active, camera, recognition_thread

    if recognition_active:
        return JSONResponse(status_code=400, content={"success": False, "status": "Recognition already running"})

    # Initialize camera only when needed
    with camera_lock:
        if camera is None or not camera.isOpened():
            if not initialize_camera():
                return JSONResponse(status_code=500, content={"success": False, "status": "Failed to initialize camera"})

    # Load face data if needed
    if not known_face_encodings or not known_face_names:
        load_known_faces()

    recognition_active = True
    
    # Start the recognition process in a background task
    asyncio.create_task(process_face_recognition())
    
    return JSONResponse(status_code=200, content={"success": True, "status": "Face recognition started"})

async def process_face_recognition():
    global recognition_active, camera

    try:
        if not known_face_encodings or not known_face_names:
            load_known_faces()

        while recognition_active:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    if not initialize_camera():
                        recognition_active = False
                        break

                ret, frame = camera.read()
                if not ret or frame is None or frame.size == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                # Make a deep copy of the frame before releasing the lock
                current_frame = frame.copy()
            
            # Process the frame outside the lock
            rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            # Draw rectangles for recognized faces when matches found
            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Process each detected face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=CONFIDENCE_THRESHOLD)
                    name = "Unknown"
                    
                    if True in matches:  # If there's at least one match
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            
                            # Update attendance log
                            update_excel_log(name)
                    
                    # Draw rectangle and name
                    cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(current_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # Add status text for better feedback
            cv2.putText(current_frame, f"Recognition Active - {len(known_face_names)} known faces", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
                
            # Convert frame to JPEG for streaming
            _, jpeg = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_binary = jpeg.tobytes()
            
            # Update the frame buffer for streaming
            update_frame_buffer(jpeg_binary)
            
            # Control frame rate
            await asyncio.sleep(DETECTION_INTERVAL)

        print("🛑 Face recognition stopped")

    except Exception as e:
        print(f"❌ Error in face recognition: {str(e)}")
        traceback.print_exc()
    finally:
        recognition_active = False

@app.post("/stop_recognition")
async def stop_recognition():
    global recognition_active, camera

    if not recognition_active:
        return {"success": False, "status": "Recognition is not running"}

    recognition_active = False
    await asyncio.sleep(0.5)
    
    # Clean up WebRTC connections
    await cleanup_video_connections()
    
    # Stop camera
    force_stop_camera()
    
    # Clear SSE connections
    active_sse_connections.clear()

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
        count = len(attendance_log.get(today, {}))
        attended_users = list(attendance_log.get(today, {}))
        
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

            # Update known_faces.json
            try:
                if os.path.exists(KNOWN_FACES_PATH):
                    with open(KNOWN_FACES_PATH, 'r') as f:
                        known_faces_data = json.load(f)
                else:
                    known_faces_data = {}
                
                # Add or update the face encoding
                known_faces_data[safe_name] = face_encoding.tolist()
                
                # Save the updated data
                with open(KNOWN_FACES_PATH, 'w') as f:
                    json.dump(known_faces_data, f, indent=4)
                
                # Reload known faces
                load_known_faces()
                
                print(f"✅ Updated known_faces.json with {safe_name}")
            except Exception as e:
                print(f"❌ Error updating known_faces.json: {str(e)}")
                return {"success": False, "status": f"Error updating face database: {str(e)}"}

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
        
        # Remove from known_faces.json
        try:
            if os.path.exists(KNOWN_FACES_PATH):
                with open(KNOWN_FACES_PATH, 'r') as f:
                    known_faces_data = json.load(f)
                
                if name in known_faces_data:
                    del known_faces_data[name]
                    
                    with open(KNOWN_FACES_PATH, 'w') as f:
                        json.dump(known_faces_data, f, indent=4)
                    
                    # Reload known faces
                    load_known_faces()
        except Exception as e:
            print(f"❌ Error updating known_faces.json: {str(e)}")
        
        return {"success": True, "status": f"User '{name}' deleted successfully"}
        
    except Exception as e:
        return {"success": False, "status": f"Server error: {str(e)}"}


@app.get("/capture_image")
async def capture_image():
    global camera
    try:
        camera_initialized = False
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera_initialized = initialize_camera()
            else:
                camera_initialized = True
                
        if not camera_initialized:
            return {"success": False, "status": "Could not initialize camera"}
        
        with camera_lock:
            # Capture multiple frames to stabilize exposure
            for _ in range(5):  
                ret, _ = camera.read()
                await asyncio.sleep(0.1)  # Replaced time.sleep() with asyncio.sleep()

            ret, frame = camera.read()
            
        if not ret or frame is None or frame.size == 0:
            return {"success": False, "status": "Failed to capture image"}
        
        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Determine bounding box color
        if len(face_locations) == 1:
            color = (0, 255, 0)  # Green for clear face
        else:
            color = (0, 0, 255)  # Red for no face or multiple faces

        # Draw bounding box around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Encode image to Base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{jpg_as_text}",
            "face_detected": len(face_locations) == 1
        }

    except Exception as e:
        print(f"❌ Error in capture_image: {str(e)}")
        return {"success": False, "status": f"Server error: {str(e)}"}



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
        face_encodings_list = []
        
        # Capture multiple images
        for i in range(MAX_CAPTURE_IMAGES):
            with camera_lock:
                # Wait a moment between captures
                await asyncio.sleep(0.5)
                ret, frame = camera.read()
                
            if not ret or frame is None or frame.size == 0:
                continue
            
            # Detect face in the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                # Skip this image if no face is detected
                continue
            
            if len(face_locations) > 1:
                # Skip this image if multiple faces are detected
                continue
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if face_encodings:
                face_encodings_list.append(face_encodings[0])
            
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
        
        # Update known_faces.json with average encoding
        if face_encodings_list:
            try:
                # Calculate average encoding
                average_encoding = np.mean(face_encodings_list, axis=0)
                
                # Load existing known faces
                if os.path.exists(KNOWN_FACES_PATH):
                    with open(KNOWN_FACES_PATH, 'r') as f:
                        known_faces_data = json.load(f)
                else:
                    known_faces_data = {}
                
                # Add or update the face encoding
                known_faces_data[safe_name] = average_encoding.tolist()
                
                # Save the updated data
                with open(KNOWN_FACES_PATH, 'w') as f:
                    json.dump(known_faces_data, f, indent=4)
                
                # Reload known faces
                load_known_faces()
                
                print(f"✅ Updated known_faces.json with {safe_name}")
                training_status = "Face model trained successfully"
            except Exception as e:
                print(f"❌ Error updating known_faces.json: {str(e)}")
                training_status = f"Error training face model: {str(e)}"
        else:
            training_status = "No faces detected in captured images"
        
        return {
            "success": face_detected_count > 0,
            "status": training_status,
            "images_captured": face_detected_count,
            "captured_images": captured_images[:5]  # Limit to first 5 images to avoid large response
        }
        
    except Exception as e:
        print(f"Error in capture_registration_images: {str(e)}")
        traceback.print_exc()
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
        uvicorn.run(app, host="127.0.0.1", port=8000)
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