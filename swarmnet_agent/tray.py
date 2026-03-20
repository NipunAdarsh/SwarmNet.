"""System tray icon and menu logic."""

import os
import json
import logging
from typing import Callable, Any
from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item
from plyer import notification
import tkinter as tk
from tkinter import messagebox

from config import DATA_FILE, APP_DIR

logger = logging.getLogger(__name__)

class TrayManager:
    def __init__(self, toggle_pause_cb: Callable, quit_cb: Callable):
        self.toggle_pause_cb = toggle_pause_cb
        self.quit_cb = quit_cb
        
        self.is_paused = False
        self.status_text = "Idle / Waiting"
        
        self.state = {
            "device_id": None,
            "user_token": "dummy_token_for_testing_12345", 
            "tasks_completed": 0,
            "hours_donated": 0.0,
            "xp": 0,
            "registered": False
        }
        self.load_state()
        
        # Internal reference to pystray icon
        self.icon = None

    def load_state(self):
        """Load state from local JSON, or keep defaults if not exists."""
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
            except Exception as e:
                logger.error(f"Failed to load state from {DATA_FILE}: {e}")

    def save_state(self):
        """Save current state to local JSON file."""
        try:
            os.makedirs(APP_DIR, exist_ok=True)
            with open(DATA_FILE, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state to {DATA_FILE}: {e}")

    def create_image(self, color):
        """Generate a simple colored circle icon."""
        width = 64
        height = 64
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        dc = ImageDraw.Draw(image)
        dc.ellipse((8, 8, width - 8, height - 8), fill=color)
        return image

    def set_icon_status(self, status: str):
        """Set the icon color and hover text based on status."""
        self.status_text = status
        
        if self.is_paused:
            color = "red"
            title = "SwarmNet - Paused"
        elif status == "Donating":
            color = "green"
            title = "SwarmNet - Donating"
        else:
            color = "yellow"
            title = f"SwarmNet - {status}"
            
        if self.icon:
            self.icon.icon = self.create_image(color)
            self.icon.title = title
            self.icon.update_menu()

    def show_stats_popup(self, icon, item):
        """Show a tkinter message box with current stats."""
        root = tk.Tk()
        root.withdraw()
        stats_msg = (
            f"Device ID: {self.state['device_id']}\n"
            f"Tasks Completed: {self.state['tasks_completed']}\n"
            f"Hours Donated: {self.state['hours_donated']:.2f}h\n"
            f"XP Earned: {self.state['xp']}"
        )
        messagebox.showinfo("SwarmNet Stats", stats_msg)
        root.destroy()

    def toggle_pause(self, icon, item):
        """Toggle the pause state."""
        self.is_paused = not self.is_paused
        self.toggle_pause_cb(self.is_paused)
        
        if self.is_paused:
            self.set_icon_status("Paused")
            self.notify("SwarmNet donation paused", "You are no longer donating compute.")
        else:
            self.set_icon_status("Idle / Waiting")
            self.notify("SwarmNet resumed", "You are now eligible to donate compute when idle.")

    def quit_app(self, icon, item):
        """Quit the application entirely."""
        logger.info("Quitting application from tray.")
        if self.icon:
            self.icon.stop()
        self.quit_cb()

    def notify(self, title: str, message: str):
        """Show a native Windows toast notification."""
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="SwarmNet Agent",
                timeout=5
            )
        except Exception as e:
            logger.error(f"Failed to show notification: {e}")

    def run(self):
        """Run the pystray blocking loop. Must be called on main thread."""
        color = "yellow"
        
        def title_text(item):
            return f"SwarmNet — {self.status_text}"
            
        menu = pystray.Menu(
            item(title_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            item('View Stats', self.show_stats_popup),
            item(lambda item: 'Resume Donation' if self.is_paused else 'Pause Donation', self.toggle_pause),
            pystray.Menu.SEPARATOR,
            item('Quit', self.quit_app)
        )
        
        self.icon = pystray.Icon("SwarmNet", self.create_image(color), "SwarmNet", menu)
        self.icon.run()
