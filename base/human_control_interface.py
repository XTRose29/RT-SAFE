import sys
import time
import math
from typing import Optional, Callable
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QGroupBox, QGridLayout, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QImage
from PIL import Image
import io
from base.rt_action_space import RTActionSpace, MOVE_TO, TURN_AROUND, WAIT
from simworld.utils.vector import Vector


class HumanControlInterface(QMainWindow):
    """Qt-based interface for human control of the RT agent."""
    
    action_selected = pyqtSignal(object)  # Emits RTActionSpace when action is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Human Control Interface - RT Agent")
        self.setGeometry(100, 100, 1200, 800)
        
        # Action selection callback
        self.action_callback: Optional[Callable] = None
        
        # Current observation data
        self.current_observation = None
        self.current_waypoints = []
        self.current_status = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left panel - Images and status
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 2)
        
        # Right panel - Action selection
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, 1)
        
    def create_left_panel(self):
        """Create the left panel with images and status information."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Images section
        images_group = QGroupBox("Agent View")
        images_layout = QVBoxLayout(images_group)
        
        # RGB Image only
        self.rgb_label = QLabel("RGB View")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setMinimumHeight(500)
        self.rgb_label.setStyleSheet("border: 1px solid gray;")
        images_layout.addWidget(self.rgb_label)
        
        layout.addWidget(images_group)
        
        # Status information
        status_group = QGroupBox("Agent Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        return panel
        
    def create_right_panel(self):
        """Create the right panel with action selection."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Action selection group
        action_group = QGroupBox("Action Selection")
        action_layout = QVBoxLayout(action_group)
        
        # Wait buttons (1s, 2s, 3s)
        self.wait_buttons = {}
        for d in [1, 2, 3]:
            btn = QPushButton(f"Wait {d}s")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff6b6b;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #ff5252;
                }
            """)
            btn.clicked.connect(lambda checked, dur=d: self.select_action(WAIT, str(dur)))
            action_layout.addWidget(btn)
            self.wait_buttons[d] = btn
        
        # Move to waypoints section
        waypoint_group = QGroupBox("Move to Waypoint")
        waypoint_layout = QGridLayout(waypoint_group)
        
        # Waypoint buttons (1-7)
        self.waypoint_buttons = {}
        for i in range(1, 8):
            btn = QPushButton(f"Waypoint {i}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4ecdc4;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #45b7b8;
                }
            """)
            btn.clicked.connect(lambda checked, wp=i: self.select_action(MOVE_TO, str(wp)))
            waypoint_layout.addWidget(btn, (i - 1) // 3, (i - 1) % 3)
            self.waypoint_buttons[i] = btn
            
        # Turn buttons (L30/L60/L90, R30/R60/R90)
        turn_style = """
            QPushButton {
                background-color: #45b7d1;
                color: white;
                font-size: 11px;
                font-weight: bold;
                padding: 6px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3a9bc1;
            }
        """
        turn_options = ['L30', 'L60', 'L90', 'R30', 'R60', 'R90']
        self.turn_buttons = {}
        for i, turn_id in enumerate(turn_options):
            btn = QPushButton(f"Turn {turn_id}")
            btn.setStyleSheet(turn_style)
            btn.clicked.connect(lambda checked, wp=turn_id: self.select_action(TURN_AROUND, wp))
            waypoint_layout.addWidget(btn, 3 + i // 3, i % 3)
            self.turn_buttons[turn_id] = btn
        
        action_layout.addWidget(waypoint_group)
        
        # Reasoning input
        reasoning_group = QGroupBox("Reasoning (Optional)")
        reasoning_layout = QVBoxLayout(reasoning_group)
        
        self.reasoning_input = QTextEdit()
        self.reasoning_input.setMaximumHeight(60)
        self.reasoning_input.setPlaceholderText("Enter your reasoning for this action...")
        reasoning_layout.addWidget(self.reasoning_input)
        
        action_layout.addWidget(reasoning_group)
        
        # Submit button
        self.submit_button = QPushButton("SUBMIT ACTION")
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.submit_button.clicked.connect(self.submit_action)
        self.submit_button.setEnabled(False)
        action_layout.addWidget(self.submit_button)
        
        layout.addWidget(action_group)
        
        # Selected action display
        selection_group = QGroupBox("Selected Action")
        selection_layout = QVBoxLayout(selection_group)
        
        self.selected_action_text = QTextEdit()
        self.selected_action_text.setMaximumHeight(100)
        self.selected_action_text.setReadOnly(True)
        selection_layout.addWidget(self.selected_action_text)
        
        layout.addWidget(selection_group)
        
        return panel
        
    def update_observation(self, observation, status_info):
        """Update the interface with new observation data."""
        self.current_observation = observation
        self.current_status = status_info
        
        # Update RGB image
        if observation and 'ego_view' in observation:
            rgb_image = observation['ego_view']
            if rgb_image:
                self.update_image_display(self.rgb_label, rgb_image)
        
        # Update waypoints
        if observation and 'waypoints' in observation:
            self.current_waypoints = observation['waypoints']
        
        # Update status information
        self.update_status_display(status_info)
        
    def update_image_display(self, label, pil_image):
        """Update a QLabel with a PIL image."""
        if pil_image is None:
            return
            
        # Convert PIL to QImage
        if pil_image.mode == 'RGB':
            rgb_image = pil_image
        else:
            rgb_image = pil_image.convert('RGB')
            
        # Convert to bytes
        buffer = io.BytesIO()
        rgb_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Create QImage from bytes
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        
    def update_status_display(self, status_info):
        """Update the status information display."""
        status_text = f"""Step: {status_info.get('step_num', 'N/A')}
Position: {status_info.get('current_position', 'N/A')}
Speed: {status_info.get('speed', 'N/A')} cm/s
Direction: {status_info.get('direction', 'N/A')}
Destination: {status_info.get('destination', 'N/A')}
Relative Distance: {status_info.get('relative_distance', 'N/A'):.2f} cm
Relative Angle: {status_info.get('relative_angle', 'N/A'):.2f} degrees
Time Spent: {status_info.get('time_spent', 'N/A'):.2f}s
Required Time: {status_info.get('required_time', 'N/A'):.2f}s

History:
{status_info.get('history', 'No history available')}"""
        
        self.status_text.setText(status_text)
        
    def select_action(self, action_type: str, action_param: str):
        """Select an action and update the display."""
        self.selected_action = {
            'action_type': action_type,
            'action_param': action_param
        }
        
        action_text = f"Action: {action_type}\nParam: {action_param}"
        self.selected_action_text.setText(action_text)
        self.submit_button.setEnabled(True)
            
    def submit_action(self):
        """Submit the selected action."""
        if not hasattr(self, 'selected_action'):
            return
            
        # Get reasoning
        reasoning = self.reasoning_input.toPlainText().strip()
        if not reasoning:
            reasoning = "Human decision"
            
        # Create RTActionSpace object
        action = RTActionSpace(
            action_type=self.selected_action['action_type'],
            action_param=self.selected_action['action_param'],
            reasoning=reasoning
        )
        
        # Emit the action
        self.action_selected.emit(action)
        
        # Reset interface
        self.submit_button.setEnabled(False)
        self.selected_action_text.clear()
        self.reasoning_input.clear()
        
    def set_action_callback(self, callback):
        """Set the callback function for when an action is selected."""
        self.action_callback = callback
        self.action_selected.connect(callback)
        
    def show_interface(self):
        """Show the interface window."""
        self.show()
        self.raise_()
        self.activateWindow()
        
        # Ensure window is not minimized and is active
        if self.windowState() & Qt.WindowMinimized:
            self.setWindowState(Qt.WindowNoState)
        
        print("Human control interface window should now be visible")

    def task_completed(self):
        """Handle task completion."""
        self.submit_button.setEnabled(False)
        for btn in self.wait_buttons.values():
            btn.setEnabled(False)
        for btn in self.waypoint_buttons.values():
            btn.setEnabled(False)
        for btn in self.turn_buttons.values():
            btn.setEnabled(False)
        
        QApplication.quit()