# Face Recognition Project

This project provides real-time face recognition capabilities using OpenCV, dlib, and face_recognition libraries.

## System Requirements

- **Python Version**: 3.9+ (Recommended 3.9.13)
- **Compiler**: Microsoft Visual C++ 14.0 or greater (Comes with Visual Studio Build Tools)
- **Operating System**: Windows 10/11 (64-bit recommended)

## Prerequisites

Before running this project, you need to install:

1. **Dlib for Windows**:
   ```bash
   git clone https://github.com/sachadee/Dlib_Windows_Python3.x-main
   cd Dlib_Windows_Python3.x-main
   pip install dlib-19.24.6-cp39-cp39-win_amd64.whl
   ```

2. **FFmpeg**:
   - Download from https://ffmpeg.org/download.html
   - Add FFmpeg to your system PATH:
     - Extract the downloaded zip file
     - Add the `bin` folder path to your system environment variables

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-recognition-project.git
   cd face-recognition-project
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Additional setup for all devices:
   - Ensure webcam access is enabled
   - For Linux/Mac users, you may need to install:
     ```bash
     sudo apt-get install build-essential cmake
     sudo apt-get install libgtk-3-dev
     sudo apt-get install libboost-all-dev
     ```

## Running the Project

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Troubleshooting

- If you encounter dlib installation issues:
  - Ensure you have Visual C++ Build Tools installed
  - Try installing CMake: `pip install cmake`
  - For ARM devices (like M1/M2 Macs), use: `pip install dlib --no-binary :all:`

- For webcam issues:
  - Check device permissions
  - Verify the correct camera index in `app.py`

## Project Structure

```
project/
├── app.py            # Main application file
├── requirements.txt  # Python dependencies
├── templates/        # HTML templates
│   ├── index.html
│   └── update.html
└── README.md         # This file
```

## Python Version Information

Your current environment is using:
- Python: 3.9.13 (as per your requirements.txt)
- Packages: See requirements.txt for complete list
- Compiler: Microsoft Visual C++ (required for dlib compilation)

## Notes

- For best performance, use a device with at least 8GB RAM
- The application may require additional permissions on mobile devices
- For deployment, consider using Docker for cross-platform compatibility
