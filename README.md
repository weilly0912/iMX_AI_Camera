# i.MX Camera GTK - AI Camera Application for NXP iMX Platforms

An AI-powered camera application with GTK3 GUI for NXP iMX embedded platforms. This application provides real-time AI inference capabilities including object detection, segmentation, face recognition, pose estimation, and more.

## Overview

i.MX Camera GTK is a comprehensive AI camera demo application designed for NXP iMX platforms. It features a modern GTK3-based graphical interface and supports various TensorFlow Lite models optimized for edge AI inference.

**Version:** 10.0  
**Last Updated:** 2025/02/06  
**Author:** Weilly Li (WPI)

## Supported Platforms

- **i.MX8MP** (BSP 5.15.x / L6.1.55)
- **i.MX93** (BSP 5.15.x)
- **i.MX95** (BSP L6.1.55)

## Features

- Real-time AI inference with TensorFlow Lite
- Multiple AI model support:
  - Object Detection (YOLOv5, YOLOv8, MobileNet SSD)
  - Semantic Segmentation (YOLOv5s)
  - Face Recognition
  - Pose Estimation
  - Hand Detection & Landmark Detection
  - Fruit Detection
  - Hard Hat Detection
  - PCB Defect Detection
  - Electronic Component Detection
  - Emotion Recognition
  - Age Estimation
- ISP (Image Signal Processor) configuration
- Dual camera support
- Multiple camera formats (YUV, RGB, MJPEG)
- Customizable window sizes and display modes
- Image and video input support
- Hardware acceleration support (NXP NPU, Ethos-U)

## Requirements

### Hardware
- NXP iMX development board (i.MX8MP, i.MX93, or i.MX95)
- Camera module (MIPI or USB)
- Display (HDMI or WXGA)

### Software Dependencies

The application requires the following Python packages and system libraries:

- Python 3.x
- GTK3 (PyGObject)
- GStreamer 1.0
- TensorFlow Lite Runtime
- OpenCV (cv2)
- NumPy
- Cairo
- PyTorch (for some post-processing)
- psutil

### System Libraries
- GStreamer plugins (including NXP-specific plugins like `imxvideoconvert_g2d`)
- Wayland or X11 display server

## Installation

### 1. Prerequisites

Ensure your system has the required dependencies installed:

```bash
# Update package list
sudo apt-get update

# Install Python and development tools
sudo apt-get install python3 python3-pip python3-dev

# Install GTK3 and GStreamer
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# Install NXP-specific GStreamer plugins (if available)
# These are typically provided in the BSP

# Install Python packages
pip3 install numpy opencv-python torch torchvision psutil

# Install TensorFlow Lite Runtime
# For ARM64/aarch64:
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0-cp39-cp39-linux_aarch64.whl

# Or build from source if the wheel is not available
```

### 2. Clone or Copy the Project

```bash
# If using git
git clone <repository-url>
cd iMX_AI_Camera

# Or extract the project files to your target directory
# Recommended location: /home/root/iMX_AI_Camera
```

### 3. Set Permissions

```bash
# Make scripts executable
chmod +x run/run.sh
chmod +x run/demo.py

# Ensure camera device access
sudo usermod -a -G video $USER
```

### 4. Verify Camera Device

```bash
# List available video devices
ls -l /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices
```

## Usage

### Basic Usage

Run the application with default settings:

```bash
cd run
python3 demo.py
```

### Command-Line Arguments

The application supports the following command-line arguments:

```bash
python3 demo.py [OPTIONS]
```

**Options:**

- `-p, --platform` (default: "1")
  - Platform selection: `0` = i.MX8MP, `1` = i.MX93, `2` = i.MX95

- `-c, --camera` (default: "0")
  - Camera device: `/dev/video0`, `/dev/video1`, etc., or numeric index (0, 1, 2...)

- `-cf, --camera_format` (default: "video/x-raw,width=1920,height=1080,framerate=30/1 !")
  - Camera format specification. Examples:
    - `video/x-raw,width=1920,height=1080,framerate=30/1 !`
    - `video/x-raw,width=1280,height=720,framerate=30/1 !`
    - `image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert !`

- `-cd, --camera_dual` (default: "0")
  - Enable dual camera display: `0` = disabled, `1` = enabled

- `-ws, --windows_set` (default: "960, 540, 1200, 720, ../data/logo/ATU_WPI_Logo_WXGA.png, ../data/style_WXGA.css")
  - Window settings: `FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY, WINDOWS_MAIN_WIDTH, WINDOWS_MAIN_HEIGHT, IMAGE_LOGO_PATH, GTK_STYLE_CSS_PATH`
  - Example for HDMI: `1472, 828, 1920, 1000, ../data/logo/WPI_NXP_Logo_300x48.png, ../data/style.css`

- `-wm, --windows_maximum` (default: "1")
  - Window maximization: `0` = normal window, `1` = maximized

### Example Commands

**For i.MX93 (OP-Gyro) with default camera:**
```bash
python3 demo.py -p 1 -c 1
```

**For i.MX8MP (OP-Killer) with USB camera:**
```bash
python3 demo.py -p 0 -c 3 -cf "video/x-raw,width=1280,height=720,framerate=30/1 !"
```

**With MJPEG format and maximized window:**
```bash
python3 demo.py -c 1 -cf "image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert !" -wm 1
```

**For WXGA display:**
```bash
python3 demo.py -c 1 -ws "960, 540, 1200, 720, ../data/logo/ATU_WPI_Logo_WXGA.png, ../data/style_WXGA.css"
```

### Using the Run Script

The `run.sh` script provides a convenient way to launch the application:

```bash
cd run
./run.sh
```

Or run in the background:

```bash
nohup ./run.sh &
```

**Note:** The script includes camera device and format configurations. Edit `run.sh` to customize settings for your platform.

### Using Compiled Binary

If a compiled binary (`demo.bin`) is available:

```bash
/home/root/iMX_AI_Camera/run/demo.bin -c 1 -cf "image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !" -wm 1
```

## Camera Device Configuration

Different platforms use different default camera devices:

- **OP-Gyro (i.MX93)**: Default is `/dev/video0` (MIPI or USB)
- **OP-Killer (i.MX8MP)**: Default is `/dev/video2` (MIPI), `/dev/video3` (USB)

## AI Models

The application supports various pre-trained TensorFlow Lite models located in `data/model/`:

- Object Detection: YOLOv5s, YOLOv8n, MobileNet SSD
- Segmentation: YOLOv5s-segmentation
- Face Recognition: FaceNet
- Pose Estimation: PoseNet
- Specialized Models: Fruit detection, Hard hat detection, PCB defect detection, etc.

Models optimized for different hardware accelerators are available:
- Standard TFLite models (`.tflite`)
- Neutron-optimized models (`model_neutron/`)
- Vela-optimized models (`model_vela/`)

## GUI Features

The GTK3 interface provides:

- Real-time video display with AI overlay
- Model selection menu
- ISP adjustment controls
- Image/video loading
- Result saving
- Platform selection
- Camera source selection
- Window controls

## Troubleshooting

### Display Issues

If you encounter display errors:

```bash
# Set display environment variable
export DISPLAY=:0
export GDK_BACKEND=wayland  # or x11 depending on your system
```

### Camera Not Detected

1. Verify camera device exists:
   ```bash
   ls -l /dev/video*
   ```

2. Check camera permissions:
   ```bash
   sudo chmod 666 /dev/video0
   ```

3. Test with v4l2-ctl:
   ```bash
   v4l2-ctl --device=/dev/video0 --all
   ```

### GStreamer Pipeline Errors

- Ensure NXP-specific GStreamer plugins are installed
- Check camera format compatibility
- Verify GStreamer version: `gst-launch-1.0 --version`

### Performance Issues

- Use optimized models (Neutron/Vela) for better performance
- Adjust camera resolution and framerate
- Enable hardware acceleration if available
- Close unnecessary applications

### Model Loading Errors

- Verify model files exist in `data/model/`
- Check model file permissions
- Ensure TensorFlow Lite runtime is correctly installed

## File Structure

```
iMX_AI_Camera/
├── data/
│   ├── model/          # AI model files (.tflite, .nb)
│   ├── label/          # Label files for models
│   ├── img/            # Sample images
│   ├── video/          # Sample videos
│   ├── logo/           # Logo images
│   ├── icon/           # UI icons
│   ├── style*.css      # GTK stylesheets
│   └── tmp/            # Temporary files and flags
├── run/
│   ├── demo.py         # Main application
│   ├── algorithm.py    # AI inference algorithms
│   ├── utils.py        # Utility functions
│   ├── settingISP.py   # ISP configuration
│   ├── run.sh          # Launch script
│   └── *.py            # Additional utility scripts
└── README.md           # This file
```

## License

Copyright (c) 2020 Freescale Semiconductor  
Copyright 2020-2024 WPI  
All Rights Reserved

This software is provided by WPI-TW "AS IS" and any expressed or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.

## Support

For issues and questions, please refer to:
- NXP iMX documentation
- WPI support channels
- Project issue tracker (if available)

## Version History

- **v1.0**: Initial release for i.MX8MP (BSP 5.15.x)
- **v2.0**: Added i.MX93 support
- **v3.0**: Added ISP and AutoRun functionality
- **v4.0**: Added OP-Gyro platform and WXGA screen support (BSP L6.1.55)
- **v5.0**: Added Cairo surface RGB888 output
- **v6.0**: Re-defined detection algorithm output
- **v7.0**: Added restart function
- **v8.0**: Updated restart function
- **v9.0**: Fixed display and crash issues (X11 related)
- **v10.0**: Updated import reference parameters, added flaginference.txt support

---

**Note:** This application is designed for embedded Linux environments on NXP iMX platforms. Some features may not work on standard desktop Linux distributions without appropriate hardware and drivers.


