from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import json
import os
import shutil
from flask import jsonify, request
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import cv2
from typing import List, Dict, Any

# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier('static/models/haarcascade_frontalface_default.xml')
import face_recognition

import io

# Initialize FastAPI app
app = FastAPI(title="Face Recognition Attendance System")

# Create necessary directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/sounds", exist_ok=True)
os.makedirs("static/models", exist_ok=True)
os.makedirs("Dataset", exist_ok=True)
os.makedirs("AttendanceData", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables
KNOWN_FACES_FILE = "known_faces.json"

# Helper functions
def initialize_known_faces_file():
    """Initialize known_faces.json file if it doesn't exist"""
    if not os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "w") as f:
            json.dump([], f)
            
def load_known_faces():
    """Load known faces from file with improved format handling"""
    file_path = KNOWN_FACES_FILE
    
    if not os.path.exists(file_path):
        # Create an empty list file if it doesn't exist
        with open(file_path, "w") as f:
            json.dump([], f)
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both dictionary and list formats
        if isinstance(data, dict):
            # Convert dictionary to list of dictionaries format
            known_faces = []
            for name, descriptor in data.items():
                known_faces.append({
                    "name": name,
                    "descriptor": descriptor
                })
            
            # Save in the new format immediately to fix the file
            save_known_faces(known_faces)
            print(f"Converted known_faces.json from dict to list format")
            return known_faces
            
        elif isinstance(data, list):
            # Already in correct format
            return data
        else:
            print(f"ERROR: known_faces is not a list or dict, it's a {type(data)}")
            # Return empty list as fallback
            return []
            
    except Exception as e:
        print(f"ERROR loading known faces: {str(e)}")
        return []

def save_known_faces(faces_data):
    """Save known faces to JSON file with backup and validation"""
    try:
        # Create backup of existing file if it exists
        if os.path.exists(KNOWN_FACES_FILE):
            backup_file = f"{KNOWN_FACES_FILE}.bak"
            import shutil
            shutil.copy2(KNOWN_FACES_FILE, backup_file)
            print(f"Created backup of known_faces at {backup_file}")
        
        # Validate data format
        if not isinstance(faces_data, list):
            print(f"WARNING: faces_data is not a list, it's a {type(faces_data)}")
            # Convert to list if it's a dictionary
            if isinstance(faces_data, dict):
                converted_data = []
                for name, descriptor in faces_data.items():
                    converted_data.append({
                        "name": name,
                        "descriptor": descriptor
                    })
                faces_data = converted_data
                print("Converted dictionary to list format")
            else:
                raise ValueError("faces_data must be a list or convertible dictionary")
        
        # Write to file
        with open(KNOWN_FACES_FILE, "w") as f:
            json.dump(faces_data, f, indent=2)
        
        print(f"Successfully saved {len(faces_data)} face records to {KNOWN_FACES_FILE}")
        return True
    except Exception as e:
        print(f"ERROR saving known faces: {str(e)}")
        
        # Try to restore from backup if save failed
        if os.path.exists(f"{KNOWN_FACES_FILE}.bak"):
            try:
                import shutil
                shutil.copy2(f"{KNOWN_FACES_FILE}.bak", KNOWN_FACES_FILE)
                print("Restored known_faces from backup after save failure")
            except Exception as restore_error:
                print(f"Failed to restore backup: {str(restore_error)}")
        
        return False

def base64_to_image(base64_string):
    """Convert base64 image string to OpenCV image"""
    # Remove the data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def get_face_encoding(img):
    """Get face encoding from image with better error handling"""
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_img)
        
        if not face_locations:
            print("No face locations detected")
            return None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if not face_encodings:
            print("Failed to get face encodings")
            return None
        
        # Return first face encoding
        return face_encodings[0].tolist()
    except Exception as e:
        print(f"Error in face encoding: {str(e)}")
        return None

def get_attendance_file_path():
    """Get the path to the current month's attendance file"""
    now = datetime.now()
    month_year = now.strftime("%B-%Y")  # e.g., "March-2025"
    file_name = f"{month_year}.xlsx"
    return os.path.join("AttendanceData", file_name)

def initialize_attendance_file(file_path):
    """Initialize the attendance Excel file if it doesn't exist"""
    if not os.path.exists(file_path):
        # Create a new DataFrame with the required columns
        df = pd.DataFrame(columns=["Date", "Name", "EntryTime", "ExitTime"])
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to Excel
        df.to_excel(file_path, index=False)
    
    return file_path

# Replace all instances of .append() with pd.concat()
# Example:

def log_attendance(name, entry_time, exit_time):
    try:
        print(f"Logging attendance for: {name}, Entry Time: {entry_time}, Exit Time: {exit_time}")
        
        # Load existing attendance data
        if os.path.exists('attendance_log.xlsx'):
            attendance_df = pd.read_excel('attendance_log.xlsx')
        else:
            attendance_df = pd.DataFrame(columns=['Name', 'Entry Time', 'Exit Time', 'Date'])
        
        # Create new entry
        new_entry = {
            'Name': name,
            'Entry Time': entry_time,
            'Exit Time': exit_time,
            'Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Use concat instead of append
        attendance_df = pd.concat([attendance_df, pd.DataFrame([new_entry])], ignore_index=True)
        
        # Save to Excel
        attendance_df.to_excel('attendance_log.xlsx', index=False)
        return True
    except Exception as e:
        print(f"Error logging attendance: {e}")
        return False
    

def log_attendance_to_excel(name, entry_time=None, exit_time=None):
    """Log attendance to Excel file with improved entry/exit handling"""
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get the attendance file path
    file_path = get_attendance_file_path()
    
    # Initialize the file if it doesn't exist
    initialize_attendance_file(file_path)
    
    # Read the existing Excel file
    df = pd.read_excel(file_path)
    
    # Convert ExitTime column to string type for comparison
    df["ExitTime"] = df["ExitTime"].astype(str)
    
    # Track whether this is an entry or exit for voice messaging
    is_entry = False
    is_exit = False
    
    # Check if there's already an entry for this person today
    today_record = df[(df["Date"] == today) & (df["Name"] == name)]
    
    if len(today_record) == 0:
        # No record for today, create a new one with entry time
        new_record = {
            "Date": today,
            "Name": name,
            "EntryTime": entry_time or datetime.now().strftime("%H:%M:%S"),
            "ExitTime": ""
        }
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        is_entry = True
        print(f"Welcome {name}, you have entered at {entry_time or datetime.now().strftime('%H:%M:%S')}")
    else:
        # Get index of today's record
        idx = today_record.index[0]
        
        # Check if exit time is already recorded
        exit_recorded = today_record["ExitTime"].iloc[0] not in ["", "nan", "None", None]
        
        if exit_recorded:
            # Both entry and exit already recorded, do nothing
            print(f"{name} already has complete attendance for today")
            # No change to is_entry or is_exit flags
        elif exit_time:
            # Update existing record with exit time
            # Important: Ensure we're updating the correct row
            df.loc[idx, "ExitTime"] = exit_time
            is_exit = True
            print(f"Goodbye {name}, you have exited at {exit_time}")
        else:
            # Default to entry if not explicitly marked as exit
            is_entry = True
            print(f"{name} already has an entry time for today")
    
    # Save the updated DataFrame
    df.to_excel(file_path, index=False)
    
    # Return entry/exit status for voice greeting
    return {"is_entry": is_entry, "is_exit": is_exit}

# Initialize the system
initialize_known_faces_file()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the index.html page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/update", response_class=HTMLResponse)
async def get_update(request: Request):
    """Serve the update.html page"""
    return templates.TemplateResponse("update.html", {"request": request})

@app.get("/api/known-faces")
async def get_known_faces():
    """Get all known faces"""
    return load_known_faces()

@app.post("/api/register-employee")
async def register_employee(request: Request):
    """Register a new employee with face images"""
    try:
        # Parse request body
        data = await request.json()
        name = data.get("name")
        images = data.get("images", [])
        
        if not name or not images:
            raise HTTPException(status_code=400, detail="Name and images are required")
        
        print(f"Processing registration for {name} with {len(images)} images")
        
        # Create employee directory in Dataset folder
        employee_dir = os.path.join("Dataset", name)
        os.makedirs(employee_dir, exist_ok=True)
        print(f"Created directory: {employee_dir}")
        
        # Process and save face images
        face_encodings = []
        
        for i, img_base64 in enumerate(images):
            try:
                # Convert base64 to image
                img = base64_to_image(img_base64)
                
                if img is None:
                    print(f"Failed to convert image {i} to OpenCV format")
                    continue
                
                # Save image to dataset folder
                img_path = os.path.join(employee_dir, f"{name}_{i}.jpg")
                cv2.imwrite(img_path, img)
                print(f"Saved image to {img_path}")
                
                # Get face encoding
                face_encoding = get_face_encoding(img)
                
                if face_encoding:
                    face_encodings.append(face_encoding)
                    print(f"Successfully encoded face in image {i}")
                else:
                    print(f"No face detected in image {i}")
            except Exception as img_error:
                print(f"Error processing image {i}: {str(img_error)}")
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No faces detected in the provided images")
        
        print(f"Processed {len(face_encodings)} face encodings")
        
        # Calculate average face encoding
        avg_encoding = np.mean(face_encodings, axis=0).tolist()
        
        # Add to known faces with improved error handling
        try:
            known_faces = load_known_faces()
            print(f"Loaded {len(known_faces)} existing known faces")
            
            # Check if employee already exists
            employee_exists = False
            for face in known_faces:
                if face["name"] == name:
                    face["descriptor"] = avg_encoding
                    employee_exists = True
                    print(f"Updated existing employee: {name}")
                    break
            
            if not employee_exists:
                known_faces.append({
                    "name": name,
                    "descriptor": avg_encoding
                })
                print(f"Added new employee: {name}")
            
            # Save updated known faces
            save_known_faces(known_faces)
            print("Saved known faces to file")
            
            # We can skip running the external training script since we've already 
            # calculated and saved the face encodings
            return {"success": True, "message": f"Employee {name} registered successfully"}
        
        except Exception as face_error:
            print(f"Error handling known faces: {str(face_error)}")
            import traceback
            print(traceback.format_exc())
            
            # Try to recover - directly create a new known_faces file with this encoding
            try:
                recovery_data = [{
                    "name": name,
                    "descriptor": avg_encoding
                }]
                with open(KNOWN_FACES_FILE, "w") as f:
                    json.dump(recovery_data, f, indent=2)
                print("Recovery: Created new known_faces file with current employee")
                return {"success": True, "message": f"Employee {name} registered with recovery method"}
            except Exception as recovery_error:
                print(f"Recovery failed: {str(recovery_error)}")
                raise HTTPException(status_code=500, detail="Failed to save employee data")
    
    except Exception as e:
        print(f"ERROR in register_employee: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log-attendance")
async def log_attendance_api(request: Request):
    """API endpoint to log attendance"""
    try:
        # Parse request body
        data = await request.json()
        name = data.get("name")
        timestamp = data.get("timestamp")
        action = data.get("action", "auto")  # 'entry', 'exit', or 'auto' (detect automatically)
        
        if not name:
            raise HTTPException(status_code=400, detail="Employee name is required")
        
        # Convert timestamp if provided, otherwise use current time
        current_time = datetime.now().strftime("%H:%M:%S")
        time_to_log = timestamp or current_time
        
        # Determine if this is an entry or exit based on current records
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = get_attendance_file_path()
        initialize_attendance_file(file_path)
        
        # Default response
        result = {
            "success": False, 
            "message": "Failed to log attendance", 
            "action": None
        }
        
        # If action is 'auto', determine what to do based on existing records
        if action == "auto":
            df = pd.read_excel(file_path)
            # Ensure proper string conversion
            if "ExitTime" in df.columns:
                df["ExitTime"] = df["ExitTime"].astype(str)
            
            # Check for today's record
            today_record = df[(df["Date"] == today) & (df["Name"] == name)]
            
            if len(today_record) == 0:
                # No record today - this is an entry
                action = "entry"
            else:
                # Check if exit time is already recorded
                exit_time = today_record["ExitTime"].iloc[0]
                exit_recorded = exit_time not in ["", "nan", "None", None, "NaT"] and pd.notna(exit_time)
                
                if exit_recorded:
                    # Both entry and exit recorded - do nothing
                    return {
                        "success": False,
                        "message": f"{name} already has complete attendance for today",
                        "action": None
                    }
                else:
                    # Entry recorded but no exit - this is an exit
                    action = "exit"
        
        # Log entry or exit as determined
        if action == "entry":
            # Add new entry record
            new_row = {
                "Date": today,
                "Name": name,
                "EntryTime": time_to_log,
                "ExitTime": ""
            }
            
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                
                # Check if there's already an entry for today
                today_entry = df[(df["Date"] == today) & (df["Name"] == name)]
                if len(today_entry) > 0:
                    # Entry already exists for today
                    return {
                        "success": False,
                        "message": f"{name} already has an entry for today",
                        "action": None
                    }
                
                # Add new entry
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Create new DataFrame with first entry
                df = pd.DataFrame([new_row])
            
            # Save updated dataframe
            df.to_excel(file_path, index=False)
            
            # Set result
            result = {
                "success": True,
                "message": f"Entry recorded for {name} at {time_to_log}",
                "action": "entry"
            }
            
        elif action == "exit":
            # Update existing record with exit time
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                
                # Find today's record
                mask = (df["Date"] == today) & (df["Name"] == name)
                matches = df[mask]
                
                if len(matches) == 0:
                    # No entry record found
                    return {
                        "success": False,
                        "message": f"No entry record found for {name} today. Cannot log exit.",
                        "action": None
                    }
                
                # Check if exit is already recorded
                exit_time = matches["ExitTime"].iloc[0]
                if exit_time and str(exit_time).strip() not in ["", "nan", "None", "NaT"] and pd.notna(exit_time):
                    # Exit already recorded
                    return {
                        "success": False,
                        "message": f"{name} already has an exit record for today",
                        "action": None
                    }
                
                # Update exit time
                index = matches.index[0]
                df.loc[index, "ExitTime"] = time_to_log
                
                # Save updated dataframe
                df.to_excel(file_path, index=False)
                
                # Set result
                result = {
                    "success": True,
                    "message": f"Exit recorded for {name} at {time_to_log}",
                    "action": "exit"
                }
            else:
                # No file exists, cannot log exit without entry
                return {
                    "success": False,
                    "message": "Cannot log exit without prior entry",
                    "action": None
                }
        
        return result
    
    except Exception as e:
        import traceback
        print(f"Error logging attendance: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/check-attendance-status")
async def check_attendance_status(name: str):
    """Check if employee has entry/exit records for today"""
    try:
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get the attendance file path
        file_path = get_attendance_file_path()
        
        # Initialize the file if it doesn't exist
        initialize_attendance_file(file_path)
        
        # Read the existing Excel file
        df = pd.read_excel(file_path)
        
        # Convert values to strings for safer comparison
        df["ExitTime"] = df["ExitTime"].astype(str)
        
        # Check if there's an entry for this person today
        today_record = df[(df["Date"] == today) & (df["Name"] == name)]
        
        has_entry = False
        has_exit = False
        
        if len(today_record) > 0:
            # Check if entry time exists and is not empty
            entry_time = today_record["EntryTime"].iloc[0]
            has_entry = entry_time is not None and str(entry_time).strip() not in ["", "nan", "None"]
            
            # Check if exit time exists and is not empty
            exit_time = today_record["ExitTime"].iloc[0]
            has_exit = exit_time is not None and str(exit_time).strip() not in ["", "nan", "None"]
        
        return {
            "name": name,
            "date": today,
            "hasEntryToday": has_entry,
            "hasExitToday": has_exit,
            "hasCompletedAttendance": has_entry and has_exit
        }
    
    except Exception as e:
        print(f"Error checking attendance status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/system-status")
async def get_system_status():
    """Get the system status including number of registered employees and recent attendance"""
    try:
        # Get known faces count
        known_faces = load_known_faces()
        employee_count = len(known_faces)
        
        # Get today's attendance count
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = get_attendance_file_path()
        
        if os.path.exists(attendance_file):
            df = pd.read_excel(attendance_file)
            today_attendance = len(df[df["Date"] == today])
        else:
            today_attendance = 0
        
        # Calculate system uptime (mock data)
        uptime = "24h"  # This would typically be calculated from system start time
        
        return {
            "employeeCount": employee_count,
            "todayAttendance": today_attendance,
            "systemUptime": uptime,
            "status": "Running"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/employee-count")
async def get_employee_count():
    """Get the total number of registered employees"""
    try:
        # Get count from known faces
        known_faces = load_known_faces()
        return {"count": len(known_faces)}
    except Exception as e:
        print(f"Error getting employee count: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/today-attendance")
async def get_today_attendance(date: str = None):
    """Get count of employees who attended today"""
    try:
        # Use provided date or default to today
        if date is None:
            today = datetime.now().strftime("%Y-%m-%d")
        else:
            today = date
            
        # Get the attendance file path
        file_path = get_attendance_file_path()
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"count": 0}
        
        # Read the attendance data
        df = pd.read_excel(file_path)
        
        # Filter for today's date
        today_records = df[df["Date"] == today]
        
        # Count unique employees
        unique_attendees = today_records["Name"].unique()
        count = len(unique_attendees)
            
        return {"count": count}
    except Exception as e:
        print(f"Error getting today's attendance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset-folders")
async def get_dataset_folders():
    """Get count of employee folders in the Dataset directory"""
    try:
        # Get list of folders in Dataset directory
        dataset_path = "Dataset"
        if not os.path.exists(dataset_path):
            return {"count": 0, "folders": []}
        
        # List all directories (exclude files and special directories)
        folders = [f for f in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, f)) 
                  and not f.startswith('.')]
        
        return {
            "count": len(folders),
            "folders": folders
        }
    except Exception as e:
        print(f"Error getting dataset folders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/employee-list")
async def get_employee_list():
    """Get a list of all registered employees"""
    try:
        known_faces = load_known_faces()
        employees = [face["name"] for face in known_faces]
        return {"employees": employees}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trigger-training")
async def trigger_training(request: Request):
    """Explicitly trigger the face training process"""
    try:
        # Get the employee name from request (optional)
        data = await request.json()
        employee_name = data.get("name", "")
        
        # Path to training script
        train_script_path = "train_face.py"
        
        if not os.path.exists(train_script_path):
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": f"Training script not found at {train_script_path}"}
            )
        
        # Run the training script in a separate process
        try:
            import subprocess
            print(f"Running training script for employee: {employee_name}")
            
            # Use Popen instead of run to prevent blocking the main thread
            process = subprocess.Popen(
                ["python", train_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # We could wait for the process to complete, but that might take too long
            # Instead, we'll return success immediately and let the process run in background
            
            return {"success": True, "message": "Training process started successfully"}
            
        except Exception as train_error:
            print(f"Training script error: {str(train_error)}")
            import traceback
            print(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Error starting training: {str(train_error)}"}
            )
            
    except Exception as e:
        print(f"Error triggering training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/models/{model_name}")
async def serve_model(model_name: str):
    """Serve face-api.js model files"""
    model_path = os.path.join("static", "models", model_name)
    if os.path.exists(model_path):
        return FileResponse(model_path)
    else:
        raise HTTPException(status_code=404, detail=f"Model file {model_name} not found")

@app.get("/sounds/{sound_name}")
async def serve_sound(sound_name: str):
    """Serve sound files"""
    sound_path = os.path.join("static", "sounds", sound_name)
    if os.path.exists(sound_path):
        return FileResponse(sound_path)
    else:
        raise HTTPException(status_code=404, detail=f"Sound file {sound_name} not found")



# Add this after the existing directories setup
@app.post("/api/save-unknown-face")
async def save_unknown_face(request: Request):
    """Save an unknown face image to a separate folder outside Dataset"""
    try:
        # Parse request body
        data = await request.json()
        image_data = data.get("imageData")
        filename = data.get("filename")
        
        if not image_data or not filename:
            raise HTTPException(status_code=400, detail="Image data and filename are required")
        
        # Create UnknownFaces directory outside Dataset if it doesn't exist
        # This addresses the requirement to store unknown faces outside Dataset folder
        unknown_dir = os.path.join("UnknownFaces")
        os.makedirs(unknown_dir, exist_ok=True)
        
        # Create a subdirectory for multiple faces if needed
        if "multi-" in filename:
            multi_dir = os.path.join(unknown_dir, "MultipleFaces")
            os.makedirs(multi_dir, exist_ok=True)
            save_dir = multi_dir
        else:
            save_dir = unknown_dir
        
        # Convert base64 to image
        # Remove the data URL prefix if present
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        # Decode base64 string
        img_data = base64.b64decode(image_data)
        
        # Save the image
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            f.write(img_data)
        
        return {"success": True, "message": "Unknown face saved successfully to separate folder"}
    
    except Exception as e:
        print(f"Error saving unknown face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)