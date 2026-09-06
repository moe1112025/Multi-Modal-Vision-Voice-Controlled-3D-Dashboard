# Multi-Modal Vision & Voice-Controlled 3D Dashboard

A Python-based real-time interactive control system combining computer vision, hand tracking, voice recognition, PyQt5, and OpenGL to create a multi-modal 3D visualization and simulation environment.

## Overview

This project demonstrates how human input can be translated into real-time digital interaction through two primary interfaces:

* Hand gestures using MediaPipe
* Voice commands using SpeechRecognition

The detected inputs are processed and mapped to camera transformations, visualization modes, and a 3D racing simulation.

## Core Technologies

* Python
* OpenCV
* MediaPipe
* NumPy
* PyQt5
* PyOpenGL
* OpenGL / GLU
* SpeechRecognition
* Threading

## System Architecture

Camera Input → OpenCV → MediaPipe Hand Tracking → Landmark Processing → Control Mapping → OpenGL Visualization

Voice Input → Speech Recognition → Command Processing → Mode Selection → Visualization

## Features

### Hand Tracking

The system tracks hand landmarks in real time and calculates:

* Hand position
* Thumb–index pinch distance
* Hand orientation
* Rotation angle
* Camera zoom
* X/Y rotation

### Voice Control

Voice commands can switch between multiple visualization modes:

* Gear
* Sphere
* Drone
* Car / Racing

### 3D Visualization

The OpenGL environment provides several interactive models:

1. Sci-Fi Gear
2. Holographic Sphere
3. Cybernetic Drone
4. 3D Racing Car

### Racing Simulation Mode

The racing mode maps hand orientation to a virtual steering mechanism, demonstrating gesture-based interaction with a simulated vehicle interface.

## Engineering Concepts

This project explores:

* Real-time computer vision
* Gesture recognition
* Human–computer interaction
* 3D graphics
* Coordinate transformation
* Real-time rendering
* Event-driven GUI programming
* Multithreaded voice processing
* Interactive simulation
* Sensor-to-interface mapping

## Project Structure

```text
project/
│
├── main.py
├── hand_tracker.py
└── README.md
```

## Installation

```bash
pip install opencv-python mediapipe numpy PyQt5 PyOpenGL PyOpenGL_accelerate SpeechRecognition
```

Additional microphone/audio dependencies may be required depending on the operating system.

## Running the Application

```bash
python main.py
```

## Controls

Voice commands:

```text
"Gear"
"Sphere"
"Drone"
"Car"
```

Hand gestures control the interactive visualization parameters.

Press `Q` where applicable to close the camera feed.

## Learning Objective

The primary objective of this project is to explore the integration of computer vision, natural human input, real-time rendering, and interactive simulation within a single Python application.

It serves as a practical demonstration of multimodal human–computer interaction and real-time vision-based control.
