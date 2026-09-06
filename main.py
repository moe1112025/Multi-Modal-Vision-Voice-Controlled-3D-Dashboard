import sys
import threading
import speech_recognition as sr
import numpy as np
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


class MultiDashboardWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom = -6.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0

        # Car Simulation Specific Variables
        self.car_steer = 0.0
        self.car_speed = 0.0

        # Smooth Damping Targets
        self.target_rot_x = 0.0
        self.target_rot_y = 0.0
        self.target_rot_z = 0.0
        self.target_zoom = -6.0
        self.target_pos_x = 0.0
        self.target_pos_y = 0.0

        # Galaxy Particle System
        np.random.seed(42)
        self.particles = np.random.uniform(-20.0, 20.0, (200, 3))

        # Active Model / Dashboard Mode (0: Gear, 1: Sphere, 2: Drone Core, 3: 3D Racing Car Sim)
        self.active_mode = 0

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.002, 0.008, 0.025, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h > 0 else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Ultra-Smooth Damping (Lerp)
        self.rot_x += (self.target_rot_x - self.rot_x) * 0.35
        self.rot_y += (self.target_rot_y - self.rot_y) * 0.35
        self.rot_z += (self.target_rot_z - self.rot_z) * 0.35
        self.zoom += (self.target_zoom - self.zoom) * 0.35
        self.pos_x += (self.target_pos_x - self.pos_x) * 0.35
        self.pos_y += (self.target_pos_y - self.pos_y) * 0.35

        # Render Galaxy Background
        self.draw_galaxy_particles()

        # Camera Transformations
        glTranslatef(self.pos_x, self.pos_y, self.zoom)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        glRotatef(self.rot_z, 0.0, 0.0, 1.0)

        # Render Selected Dashboard Item / Game Simulator
        if self.active_mode == 0:
            self.draw_sci_fi_gear()
        elif self.active_mode == 1:
            self.draw_holographic_sphere()
        elif self.active_mode == 2:
            self.draw_cybernetic_drone()
        elif self.active_mode == 3:
            self.draw_3d_racing_car()

    def draw_galaxy_particles(self):
        glPointSize(2.0)
        glColor4f(0.2, 0.7, 1.0, 0.5)
        glBegin(GL_POINTS)
        for p in self.particles:
            glVertex3f(p[0], p[1], p[2])
        glEnd()

    def draw_sci_fi_gear(self):
        glColor4f(0.0, 0.9, 1.0, 0.9)
        glLineWidth(2.5)
        num_teeth, inner_r, outer_r, depth = 16, 1.4, 2.1, 0.5
        for d in [-depth, depth]:
            glBegin(GL_LINE_LOOP)
            for i in range(num_teeth * 4):
                angle = 2 * np.pi * i / (num_teeth * 4)
                r = outer_r if (i // 2) % 2 == 0 else inner_r
                glVertex3f(r * np.cos(angle), r * np.sin(angle), d)
            glEnd()

    def draw_holographic_sphere(self):
        glColor4f(0.0, 1.0, 0.4, 0.85)
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_LINE)
        gluSphere(quad, 1.8, 16, 16)

    def draw_cybernetic_drone(self):
        glColor4f(1.0, 0.4, 0.8, 0.9)
        glLineWidth(2.0)
        r = 1.5
        glBegin(GL_LINES)
        edges = [(-r, -r, -r, r, -r, -r), (r, -r, -r, r, r, -r), (r, r, -r, -r, r, -r), (-r, r, -r, -r, -r, -r),
                 (-r, -r, r, r, -r, r), (r, -r, r, r, r, r), (r, r, r, -r, r, r), (-r, r, r, -r, -r, r),
                 (-r, -r, -r, -r, -r, r), (r, -r, -r, r, -r, r), (r, r, -r, r, r, r), (-r, r, -r, -r, r, r)]
        for e in edges:
            glVertex3f(e[0], e[1], e[2])
            glVertex3f(e[3], e[4], e[5])
        glEnd()

    def draw_3d_racing_car(self):
        # 3D Wireframe Racing Car Simulator Model
        glColor4f(1.0, 0.6, 0.0, 0.9)  # Neon Orange Racing Theme
        glLineWidth(2.5)

        # Car Body Box
        glBegin(GL_LINES)
        car_edges = [
            (-1.2, -0.4, -0.6, 1.2, -0.4, -0.6), (1.2, -0.4, -0.6, 1.2, 0.4, -0.6), (1.2, 0.4, -0.6, -1.2, 0.4, -0.6),
            (-1.2, 0.4, -0.6, -1.2, -0.4, -0.6),
            (-1.2, -0.4, 0.6, 1.2, -0.4, 0.6), (1.2, -0.4, 0.6, 1.2, 0.4, 0.6), (1.2, 0.4, 0.6, -1.2, 0.4, 0.6),
            (-1.2, 0.4, 0.6, -1.2, -0.4, 0.6),
            (-1.2, -0.4, -0.6, -1.2, -0.4, 0.6), (1.2, -0.4, -0.6, 1.2, -0.4, 0.6), (1.2, 0.4, -0.6, 1.2, 0.4, 0.6),
            (-1.2, 0.4, -0.6, -1.2, 0.4, 0.6)
        ]
        for e in car_edges:
            glVertex3f(e[0], e[1], e[2])
            glVertex3f(e[3], e[4], e[5])
        glEnd()

    def set_state(self, rx, ry, rz, zoom, px, py, mode):
        self.target_rot_x = rx
        self.target_rot_y = ry
        self.target_rot_z = rz
        self.target_zoom = zoom
        self.target_pos_x = px
        self.target_pos_y = py
        self.active_mode = mode
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Dashboard & 3D Car Racing Sim Controller")
        self.resize(1400, 800)

        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)

        self.gl_widget = MultiDashboardWidget(self)
        layout.addWidget(self.gl_widget)
        self.setCentralWidget(central_widget)

        # MediaPipe Hands Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Voice Recognition Assistant Thread
        self.voice_mode = 0
        self.init_voice_assistant()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def init_voice_assistant(self):
        def listen_loop():
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    text = recognizer.recognize_google(audio).lower()
                    if "gear" in text:
                        self.voice_mode = 0
                    elif "sphere" in text:
                        self.voice_mode = 1
                    elif "drone" in text:
                        self.voice_mode = 2
                    elif "car" in text or "game" in text or "racing" in text:
                        self.voice_mode = 3
                except:
                    pass

        threading.Thread(target=listen_loop, daemon=True).start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)

        rx, ry, rz, zoom, px, py = 0.0, 0.0, 0.0, -6.0, 0.0, 0.0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            thumb = hand.landmark[4]
            index = hand.landmark[8]
            wrist = hand.landmark[0]
            pinky_mcp = hand.landmark[17]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            pinch_dist = np.hypot(x2 - x1, y2 - y1)

            if self.voice_mode == 3:
                # 3D Car Racing Game Steering Mode (Steering Wheel Simulation)
                # Left/Right hand tilt controls car steering rotation (rz)
                angle_rad = np.arctan2(pinky_mcp.y - wrist.y, pinky_mcp.x - wrist.x)
                rz = np.degrees(angle_rad) * 1.5
                zoom = -5.0
                rx = 20.0  # Fixed tilted perspective for racing view
                ry = rz * 0.5
            else:
                # Standard Dashboard Item Control Mode
                zoom = np.interp(pinch_dist, [30, 220], [-11.0, -3.0])
                rx = np.interp(y2, [0, h], [75, -75])
                ry = np.interp(x2, [0, w], [-75, 75])
                angle_rad = np.arctan2(pinky_mcp.y - wrist.y, pinky_mcp.x - wrist.x)
                rz = np.degrees(angle_rad) * 0.8

            cv2.circle(frame, (x1, y1), 6, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 6, (0, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # HUD Dashboard Overlay Info
        cv2.rectangle(frame, (30, 30), (w - 30, h - 30), (0, 255, 200), 2)
        cv2.putText(frame, "MULTI-DASHBOARD & CAR SIM SYSTEM", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200),
                    2)
        cv2.putText(frame,
                    f"ACTIVE ITEM MODE: {['GEAR CORE', 'HOLOGRAPHIC SPHERE', 'CYBERNETIC DRONE', '3D RACING CAR SIM'][self.voice_mode]}",
                    (45, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        cv2.putText(frame, "Say 'Gear', 'Sphere', 'Drone', or 'Car' to switch items!", (45, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        self.gl_widget.set_state(rx, ry, rz, zoom, px, py, self.voice_mode)
        cv2.imshow("Dashboard Control Feed", frame)

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())