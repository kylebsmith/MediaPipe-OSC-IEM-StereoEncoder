# MediaPipe-OSC-IEM-StereoEncoder
This repository provides code for real-time hand tracking and spatial audio control. Using a camera feed, it tracks a user’s hands, normalizes their position data, and sends the values via OSC. These OSC messages are designed for integration with the IEM VST3 plugin suite, which enables intuitive spatialization of audio in 3D environments.

# MediaPipe-OSC-IEM

dual hand tracking → OSC out → IEM StereoEncoder in Reaper

## Install
    pip install numpy opencv-python mediapipe python-osc

## Defaults
- right hand → **port 9002**  
- left hand  → **port 9003**  
(use `--port-right` / `--port-left` to override)

## Handedness
- `--hand right` / `--hand left` / `--hand both`  
- mirror is **on by default** (so it matches POV)  
- use `--no-mirror` for raw camera coords

## Run Example
    python MediaPipe-OSC-IEM.py --hand both --map polar

## What it does
- opens cam  
- tracks hands (bbox + centroid overlay)  
- maps x,y → azimuth / elevation  
- openness/speed/radius → width  
- sends OSC straight into IEM StereoEncoder  

## Included
- `MediaPipe-OSC-IEM.py`  
- `KSmith-MediaPipe-OSC-IEM-Demo.mp4` (screen capture w/ cam + IEM VST3 Reaper plugin moving)
