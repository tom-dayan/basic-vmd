# Video Analytics Pipeline

## Overview

A modular, multi-process video analytics pipeline implemented in Python, designed for efficient motion detection and real-time video processing.

## Project Structure

- `streamer.py`: Handles video frame acquisition
- `detector.py`: Implements motion detection logic
- `renderer.py`: Manages frame annotation and display
- `main.py`: Orchestrates the multiprocessing pipeline

## Prerequisites

- Python 3.8+
- OpenCV
- Required dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic execution (full or short video flag)
python main.py --video path/to/video.mp4
python main.py -v path/to/video.mp4

# Optional motion blur (full or short blur flag)
python main.py --video path/to/video.mp4 --blur
python main.py -v path/to/video.mp4 -b
```

## Key Features

- Multi-process video analytics
- Real-time motion detection
- Optional region blurring
- Configurable rendering options

### Renderer Configurations

Customize in `renderer.py`:
- Bounding box color
- Timestamp styling
- Blur kernel size

## Termination

- Press `q` to gracefully stop video processing
