# run_system.py
from driver_drowsiness import DrowsinessDetector

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()