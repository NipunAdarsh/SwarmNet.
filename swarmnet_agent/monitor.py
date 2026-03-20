"""CPU idle detection using psutil."""

import logging
import psutil

from config import IDLE_CPU_THRESHOLD, IDLE_DURATION_REQUIRED, MONITOR_INTERVAL

logger = logging.getLogger(__name__)

class CPUMonitor:
    def __init__(self):
        self.idle_seconds = 0
        
    def check_idle(self) -> bool:
        """
        Check if the CPU has been idle for the required duration.
        Returns True if idle, False otherwise.
        """
        try:
            # Get CPU percentage over a 1 second sample
            cpu_percent = psutil.cpu_percent(interval=1.0)
            
            if cpu_percent < IDLE_CPU_THRESHOLD:
                self.idle_seconds += (MONITOR_INTERVAL)
                logger.debug(f"System idle: CPU at {cpu_percent}%. Intending to reach {IDLE_DURATION_REQUIRED}s. Current: {self.idle_seconds}s")
            else:
                self.idle_seconds = 0
                logger.debug(f"System active: CPU at {cpu_percent}%. Resetting idle timer.")
                
            return self.idle_seconds >= IDLE_DURATION_REQUIRED
        except Exception as e:
            logger.error(f"Error checking CPU usage: {e}")
            return False

    def reset(self):
        """Reset the idle timer."""
        self.idle_seconds = 0
