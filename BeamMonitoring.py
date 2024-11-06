import cv2
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import time
from threading import Thread, Lock
from queue import Queue
import sys
import logging
from typing import Optional, Tuple, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('beam_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class BeamMonitorException(Exception):
    """Base exception class for beam monitoring errors"""
    pass

class CameraError(BeamMonitorException):
    """Exception raised for camera-related errors"""
    pass

class FittingError(BeamMonitorException):
    """Exception raised for Gaussian fitting errors"""
    pass

class BeamAnalyzer:
    def __init__(self):
        """Initialize the 2D Gaussian fitting class"""
        self.lock = Lock()
        self.centers_queue = Queue(maxsize=100)
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
        """2D Gaussian function"""
        x, y = xy
        exp_term = -((x - x0)**2 / (2 * sigma_x**2) + 
                     (y - y0)**2 / (2 * sigma_y**2))
        return offset + amplitude * np.exp(exp_term)

    def fit_gaussian(self, image: np.ndarray) -> Optional[dict]:
        """
        Fit a 2D Gaussian to the image
        
        Args:
            image: Input grayscale image as numpy array
            
        Returns:
            Dictionary containing fit parameters or None if fit fails
            
        Raises:
            FittingError: If the Gaussian fitting fails
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image input")

            # Create x and y indices
            y, x = np.indices(image.shape)
            xy = (x.ravel(), y.ravel())
            
            # Initial guesses for parameters
            max_intensity = np.max(image)
            center_y, center_x = np.unravel_index(np.argmax(image), image.shape)
            
            initial_guess = [
                max_intensity,  # amplitude
                center_x,       # x0
                center_y,       # y0
                20.0,          # sigma_x
                20.0,          # sigma_y
                0.0            # offset
            ]
            
            # Perform the fit
            popt, pcov = curve_fit(self.gaussian_2d, xy, image.ravel(), 
                                 p0=initial_guess, maxfev=1000)
            
            # Validate fit results
            if not np.all(np.isfinite(popt)):
                raise FittingError("Fit resulted in non-finite parameters")
            
            return {
                'x': float(popt[1]), 
                'y': float(popt[2]),
                'sigma_x': float(popt[3]), 
                'sigma_y': float(popt[4]),
                'amplitude': float(popt[0]),
                'offset': float(popt[5])
            }
            
        except Exception as e:
            self.logger.error(f"Gaussian fitting failed: {str(e)}")
            raise FittingError(f"Failed to fit Gaussian: {str(e)}") from e

class BeamMonitor:
    def __init__(self, camera_index: int = 0, save_interval: int = 30, 
                 retry_interval: int = 5):
        """
        Initialize the beam monitoring system
        
        Args:
            camera_index: Index of the camera to use
            save_interval: Interval between saves in seconds
            retry_interval: Interval between camera connection retries in seconds
        """
        self.camera_index = camera_index
        self.save_interval = save_interval
        self.retry_interval = retry_interval
        self.analyzer = BeamAnalyzer()
        self.running = False
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        try:
            self.base_dir = Path("beam_monitor_data")
            self.image_dir = self.base_dir / "images"
            self.data_dir = self.base_dir / "data"
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise BeamMonitorException(f"Failed to create directories: {str(e)}")
        
        # Initialize data storage
        self.positions_df = pd.DataFrame(columns=['timestamp', 'x', 'y'])
        self.last_save_time = time.time()

    @staticmethod
    def list_available_cameras(max_cameras: int = 10) -> List[int]:
        """
        List all available camera indices
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            List of available camera indices
        """
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras

    def connect_camera(self) -> bool:
        """
        Attempt to connect to the camera
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Camera connection error: {str(e)}")
            return False

    def start(self):
        """
        Start the monitoring system
        
        Raises:
            CameraError: If camera initialization fails
        """
        # Check available cameras
        available_cameras = self.list_available_cameras()
        if not available_cameras:
            raise CameraError("No cameras available")
        
        if self.camera_index not in available_cameras:
            available_str = ", ".join(map(str, available_cameras))
            raise CameraError(
                f"Camera index {self.camera_index} not available. "
                f"Available cameras: {available_str}"
            )
        
        # Try to connect to camera
        if not self.connect_camera():
            raise CameraError(f"Failed to initialize camera {self.camera_index}")
        
        self.running = True
        self.logger.info("Starting beam monitor")
        
        # Start the monitoring thread
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # Allow program to exit if thread is running
        self.monitor_thread.start()
        
        # Start the visualization
        self._start_visualization()
        
    def stop(self):
        """Stop the monitoring system"""
        self.logger.info("Stopping beam monitor")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)  # Wait up to 5 seconds
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        self._save_final_data()
        
    def _monitor_loop(self):
        """Main monitoring loop with error handling and recovery"""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.running:
            try:
                if not self.cap.isOpened():
                    raise CameraError("Camera connection lost")
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise CameraError("Failed to capture frame")
                    
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Analyze the beam
                result = self.analyzer.fit_gaussian(gray)
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                timestamp = datetime.now()
                
                # Save data
                new_row = pd.DataFrame({
                    'timestamp': [timestamp],
                    'x': [result['x']],
                    'y': [result['y']]
                })
                self.positions_df = pd.concat([self.positions_df, new_row], 
                                           ignore_index=True)
                
                # Add to visualization queue
                self.analyzer.centers_queue.put((result['x'], result['y']))
                
                # Save image if interval has elapsed
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self._save_data(frame, timestamp)
                    self.last_save_time = current_time
                    
            except CameraError as e:
                consecutive_failures += 1
                self.logger.error(f"Camera error: {str(e)}")
                
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error("Too many consecutive failures, attempting camera reconnection")
                    if not self.connect_camera():
                        self.logger.error("Camera reconnection failed, waiting before retry")
                        time.sleep(self.retry_interval)
                        
            except FittingError as e:
                self.logger.warning(f"Fitting error: {str(e)}")
                # Continue monitoring even if fitting fails
                
            except Exception as e:
                self.logger.error(f"Unexpected error in monitor loop: {str(e)}")
                time.sleep(1)  # Prevent rapid-fire errors
                
    def _save_data(self, frame: np.ndarray, timestamp: datetime):
        """
        Save the current frame and update the data file
        
        Args:
            frame: Current camera frame
            timestamp: Current timestamp
        """
        try:
            # Save image
            image_filename = self.image_dir / f"beam_{timestamp:%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(image_filename), frame)
            
            # Save position data
            self._save_positions()
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")

    def _save_positions(self):
        """Save position data to CSV"""
        try:
            data_filename = self.data_dir / "beam_positions.csv"
            self.positions_df.to_csv(data_filename, index=False)
        except Exception as e:
            self.logger.error(f"Error saving positions data: {str(e)}")

    def _save_final_data(self):
        """Save final data before shutdown"""
        try:
            self._save_positions()
            self.logger.info("Final data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving final data: {str(e)}")
        
    def _start_visualization(self):
        """Initialize and start the real-time visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Initialize scatter plot
            scatter = ax1.scatter([], [], c='b', alpha=0.5)
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title('Beam Center Position')
            
            # Initialize time series
            line_x, = ax2.plot([], [], 'b-', label='X Position')
            line_y, = ax2.plot([], [], 'r-', label='Y Position')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Position')
            ax2.set_title('Position vs Time')
            ax2.legend()
            
            def update(frame):
                try:
                    # Update scatter plot
                    positions = []
                    while not self.analyzer.centers_queue.empty():
                        positions.append(self.analyzer.centers_queue.get())
                    
                    if positions:
                        x_pos, y_pos = zip(*positions)
                        scatter.set_offsets(np.c_[x_pos, y_pos])
                        
                        # Update time series
                        if len(self.positions_df) > 0:
                            times = np.arange(len(self.positions_df))
                            line_x.set_data(times, self.positions_df['x'])
                            line_y.set_data(times, self.positions_df['y'])
                            
                            # Adjust axes
                            ax2.relim()
                            ax2.autoscale_view()
                    
                except Exception as e:
                    self.logger.error(f"Error in visualization update: {str(e)}")
                
                return scatter, line_x, line_y
            
            ani = FuncAnimation(fig, update, interval=100)
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error starting visualization: {str(e)}")
            raise

def main():
    try:
        # Check for available cameras first
        available_cameras = BeamMonitor.list_available_cameras()
        if not available_cameras:
            print("No cameras found. Please connect a camera and try again.")
            return
            
        print(f"Available cameras: {available_cameras}")
        camera_index = available_cameras[0]  # Use first available camera
        
        # Initialize and start the beam monitor
        monitor = BeamMonitor(camera_index=camera_index, save_interval=30)
        monitor.start()
        
    except KeyboardInterrupt:
        print("\nStopping beam monitor...")
        if 'monitor' in locals():
            monitor.stop()
    except CameraError as e:
        print(f"Camera error: {str(e)}")
        print("Please check camera connection and permissions")
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'monitor' in locals():
            monitor.stop()
    finally:
        print("Beam monitor shutdown complete")

if __name__ == "__main__":
    main()