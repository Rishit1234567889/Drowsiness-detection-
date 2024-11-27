import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
from datetime import datetime
import pygame
import csv
import os
from threading import Thread
import json
from pathlib import Path
import webbrowser

class DrowsinessDetector:
    def __init__(self):
        # Video capture initialization
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture device")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Could not find {predictor_path}. Please download it from dlib's website.")
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize audio
        pygame.mixer.init()
        
        # Thresholds and constants
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.25
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 3
        self.YAWN_THRESHOLD = 0.6
        self.YAWN_CONSEC_FRAMES = 15
        self.MIN_BLINK_TIME_GAP = 0.1  # seconds
        self.ALARM_COOLDOWN = 3  # seconds
        
        # Initialize all state variables
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.sleep_frames = 0
        self.drowsy_frames = 0
        self.active_frames = 0
        self.yawn_frames = 0
        self.status = "Active"
        self.color = (0, 255, 0)
        self.is_yawning = False
        
        # Initialize all timing variables
        self.start_time = datetime.now()
        self.last_blink_time = time.time()
        self.last_alert_time = datetime.now()  # Added missing initialization
        
        # Initialize logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / f"drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.initialize_log()
        
        # Initialize alarm sound
        try:
            self.alarm_sound = pygame.mixer.Sound("alarm.wav")
        except Exception as e:
            print(f"Warning: Could not load alarm.wav ({str(e)}). Audio alerts will be disabled.")
            self.alarm_sound = None
        
        # Initialize statistics
        self.stats = {
            "total_sleep_incidents": 0,
            "total_drowsy_incidents": 0,
            "total_yawns": 0,
            "total_blinks": 0,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alert_times": [],
            "yawn_stages": []
        }
        
        # Initialize dashboard
        self.dashboard_thread = None
        self.start_dashboard()

    def start_dashboard(self):
        try:
            from dashboard import start_dashboard
            self.dashboard_thread = start_dashboard(str(self.log_file))
        except Exception as e:
            print(f"Failed to start dashboard: {e}")

    def play_alarm(self):
        """Play the alarm sound if available and enough time has passed since last alert"""
        current_time = datetime.now()
        if self.alarm_sound and (current_time - self.last_alert_time).seconds >= self.ALARM_COOLDOWN:
            try:
                self.alarm_sound.play()
                self.last_alert_time = current_time
                self.stats["alert_times"].append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as e:
                print(f"Error playing alarm: {e}")

    def log_status(self, ear, mar):
        """Log the current status and metrics to CSV"""
        try:
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.status,
                    f"{ear:.3f}",
                    f"{mar:.3f}",
                    self.TOTAL_BLINKS,
                    self.stats["total_yawns"]
                ])
        except Exception as e:
            print(f"Error logging status: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.save_statistics()
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession statistics saved to {self.log_dir}")
            print(f"Dashboard data saved to {self.log_file}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def run(self):
        """Main loop to run the drowsiness detection"""
        try:
            while True:
                frame, face_frame = self.process_frame()
                if frame is None:
                    break

                cv2.imshow("Drowsiness Detection", frame)
                if face_frame is not None:
                    cv2.imshow("Face Landmarks", face_frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break

        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    # [Rest of the methods remain the same as before]

    def compute_ear(self, eye_points):
        """Compute the eye aspect ratio with improved accuracy"""
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return ear

    def compute_mar(self, mouth_points):
        """Compute the mouth aspect ratio with improved accuracy and vertical mouth opening"""
    # Vertical mouth points
        top_lip_points = mouth_points[2:5]
        bottom_lip_points = mouth_points[8:11]
        
        # Horizontal mouth points
        mouth_width_points = [mouth_points[0], mouth_points[6]]
        
        # Compute vertical mouth opening
        vertical_opening = np.mean([np.linalg.norm(np.subtract(top_lip, bottom_lip)) 
                                    for top_lip, bottom_lip in zip(top_lip_points, bottom_lip_points)])
        
        # Compute mouth width
        mouth_width = np.linalg.norm(np.subtract(mouth_width_points[0], mouth_width_points[1]))
        
        # More nuanced mouth aspect ratio calculation
        if mouth_width == 0:
            return 0.0
        
        mar = vertical_opening / mouth_width
        return mar

    def detect_blink(self, ear):
        """Improved blink detection with timing constraints"""
        current_time = time.time()
        
        if ear < self.EYE_ASPECT_RATIO_THRESHOLD:
            self.COUNTER += 1
        else:
            if self.COUNTER >= self.EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if current_time - self.last_blink_time > self.MIN_BLINK_TIME_GAP:
                    self.TOTAL_BLINKS += 1
                    self.last_blink_time = current_time
            self.COUNTER = 0

    def detect_yawn(self, mar):
        """Enhanced yawn detection with multiple stages of yawning"""
        # Adjust these thresholds based on your specific use case
        YAWN_THRESHOLD_MILD = 0.4   # Minor mouth opening
        YAWN_THRESHOLD_MODERATE = 0.6  # Significant mouth opening
        YAWN_THRESHOLD_SEVERE = 0.8  # Extreme mouth opening
        
        YAWN_FRAMES_MILD = 10
        YAWN_FRAMES_MODERATE = 7
        YAWN_FRAMES_SEVERE = 5
        
        # Categorize yawn intensity
        if mar > YAWN_THRESHOLD_SEVERE:
            # Extreme yawn
            self.yawn_frames += 2
            current_yawn_stage = "Severe Yawn"
        elif mar > YAWN_THRESHOLD_MODERATE:
            # Moderate yawn
            self.yawn_frames += 1
            current_yawn_stage = "Moderate Yawn"
        elif mar > YAWN_THRESHOLD_MILD:
            # Mild yawn
            self.yawn_frames += 0.5
            current_yawn_stage = "Mild Yawn"
        else:
            # Reset yawn tracking
            self.yawn_frames = max(0, self.yawn_frames - 0.2)
            current_yawn_stage = None
        
        # Determine yawn detection
        if self.yawn_frames >= YAWN_FRAMES_MILD:
            if not self.is_yawning:
                self.is_yawning = True
                self.stats["total_yawns"] += 1
                self.stats["yawn_stages"].append(current_yawn_stage)
                
                # Optional: Trigger more specific alerts based on yawn intensity
                if current_yawn_stage == "Severe Yawn":
                    self.play_alarm()  # More aggressive intervention for severe yawns
            
            return True
        else:
            self.is_yawning = False
            return False

    def process_frame(self):
        """Process a single frame with improved detection logic and yawn tracking"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        face_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        # Yawn detection constants
        YAWN_THRESHOLD_MILD = 0.4
        YAWN_THRESHOLD_MODERATE = 0.6
        YAWN_THRESHOLD_SEVERE = 0.8

        if len(faces) > 0:
            for face in faces:
                landmarks = face_utils.shape_to_np(self.predictor(gray, face))

                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                mouth = landmarks[48:68]

                left_ear = self.compute_ear(left_eye)
                right_ear = self.compute_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = self.compute_mar(mouth)

                self.detect_blink(ear)
                is_yawning = self.detect_yawn(mar)

                # Determine status and color based on detection
                if ear < self.EYE_ASPECT_RATIO_THRESHOLD and self.COUNTER >= self.EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    self.status = "SLEEPING !!!"
                    self.color = (0, 0, 255)
                    self.play_alarm()
                elif is_yawning:
                    if mar > YAWN_THRESHOLD_SEVERE:
                        self.status = "Severe Yawn !!!"
                        self.color = (0, 0, 255)  # Red for severe yawn
                    elif mar > YAWN_THRESHOLD_MODERATE:
                        self.status = "Moderate Yawn !"
                        self.color = (0, 165, 255)  # Orange for moderate yawn
                    else:
                        self.status = "Mild Yawn"
                        self.color = (0, 255, 255)  # Yellow for mild yawn
                    
                    # Play alarm for moderate and severe yawns
                    if mar > YAWN_THRESHOLD_MODERATE:
                        self.play_alarm()
                else:
                    self.status = "Active"
                    self.color = (0, 255, 0)

                # Draw contours for eyes and mouth
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)

                cv2.drawContours(face_frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(face_frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(face_frame, [mouth_hull], -1, (0, 255, 0), 1)

                # Detailed text overlays
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f} - {self.status}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Blinks: {self.TOTAL_BLINKS}", (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Yawns: {self.stats['total_yawns']}", (300, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, self.status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

                # Log the detailed status
                self.log_status(ear, mar)
        else:
            cv2.putText(frame, "No Face Detected", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame, face_frame

    def initialize_log(self):
        """Initialize the CSV log file with headers"""
        with open(self.log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", 
                "Status", 
                "Eye_Aspect_Ratio", 
                "Mouth_Aspect_Ratio",
                "Blink_Count",
                "Yawn_Count"
            ])


    def save_statistics(self):
        """Save session statistics to JSON"""
        stats_file = self.log_dir / f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
if __name__ == "__main__":
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"Fatal error: {e}")
