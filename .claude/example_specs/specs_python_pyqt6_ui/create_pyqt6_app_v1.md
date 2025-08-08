# PyQt6 Desktop Application Specification v1

## Core Challenge
Create a **modern PyQt6 desktop application** that demonstrates professional GUI development practices while solving a specific user need. Focus on creating polished, responsive desktop interfaces that feel native and intuitive.

## Output Requirements

**File Naming**: `pyqt6_app_[iteration_number].py`

**Content Structure**: Complete PyQt6 application in a single Python file
```python
#!/usr/bin/env python3
"""
[Application Name] - [Brief Description]
A modern PyQt6 desktop application that [key purpose]
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    # Import all needed widgets
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QPalette, QColor

class MainWindow(QMainWindow):
    """Main application window with [key features]"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("[Application Name]")
        self.setMinimumSize(800, 600)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create UI components
        # Apply modern styling
        # Set up layouts
        
    def setup_connections(self):
        """Connect signals and slots"""
        pass

def main():
    app = QApplication(sys.argv)
    
    # Apply modern application styling
    app.setStyle("Fusion")  # or custom style
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

## Application Categories

### **Productivity Tools**
- **Task Manager**: Todo lists with categories, priorities, due dates, and notifications
- **Note Taking**: Rich text editor with organization, search, and export features
- **Time Tracker**: Project time tracking with reporting and analytics
- **Password Manager**: Secure storage with encryption and password generation
- **File Organizer**: Batch file operations with rules and automation

### **Data & Analytics**
- **Data Visualizer**: Interactive charts and graphs with real-time updates
- **Database Browser**: SQLite/PostgreSQL viewer with query builder
- **Log Analyzer**: Parse and visualize log files with filtering
- **CSV Editor**: Advanced spreadsheet functionality with formulas
- **JSON Explorer**: Tree view editor with validation and formatting

### **Creative Tools**
- **Image Viewer**: Gallery with editing capabilities and batch operations
- **Color Picker**: Advanced color selection with palettes and history
- **Font Manager**: Preview and organize system fonts
- **Icon Generator**: Create icons in multiple sizes and formats
- **ASCII Art Creator**: Convert images to ASCII with customization

### **System Utilities**
- **System Monitor**: CPU, memory, disk usage with graphs
- **Network Scanner**: Discover devices and services on network
- **Process Manager**: Enhanced task manager with detailed info
- **Backup Tool**: Scheduled backups with compression
- **Settings Manager**: Application preferences with profiles

## Design Principles

### **Modern UI/UX**
- **Clean Interface**: Minimalist design with focus on content
- **Consistent Styling**: Unified color scheme and typography
- **Responsive Layouts**: Adapt to window resizing gracefully
- **Dark Mode Support**: Toggle between light and dark themes
- **Smooth Animations**: Subtle transitions and feedback

### **PyQt6 Best Practices**
- **Signal/Slot Architecture**: Proper event handling and communication
- **MVC Pattern**: Separate data, presentation, and logic
- **Thread Safety**: Long operations in separate threads
- **Resource Management**: Proper cleanup and memory handling
- **Custom Widgets**: Extend base widgets for reusability

### **Professional Features**
- **Keyboard Shortcuts**: Comprehensive keyboard navigation
- **Context Menus**: Right-click functionality where appropriate
- **Drag and Drop**: Intuitive file and data handling
- **Undo/Redo**: Command pattern for reversible actions
- **Preferences**: Persistent settings with QSettings

## Technical Implementation

### **Core Components**
```python
# Custom styled widgets
class StyledButton(QPushButton):
    """Modern styled button with hover effects"""
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.setStyleSheet(self.get_style(primary))
        
# Responsive layouts
class ResponsiveLayout(QVBoxLayout):
    """Layout that adapts to window size"""
    def __init__(self):
        super().__init__()
        self.setup_responsive_behavior()

# Data models
class CustomTableModel(QAbstractTableModel):
    """Efficient data model for large datasets"""
    def __init__(self, data):
        super().__init__()
        self._data = data
```

### **Styling Approach**
```python
# Modern dark theme
DARK_STYLE = """
QMainWindow {
    background-color: #2b2b2b;
    color: #ffffff;
}
QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    padding: 8px 16px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #484848;
}
"""

# Apply theme
app.setStyleSheet(DARK_STYLE)
```

### **Threading Pattern**
```python
class WorkerThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    
    def run(self):
        # Perform long-running operation
        for i in range(100):
            self.progress.emit(i)
            # Do work
        self.result.emit(data)
```

## Quality Standards

### **Code Quality**
- **PEP 8 Compliance**: Follow Python style guidelines
- **Type Hints**: Use type annotations for clarity
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Graceful exception handling with user feedback
- **Logging**: Structured logging for debugging

### **Performance**
- **Efficient Updates**: Minimize redraws and calculations
- **Lazy Loading**: Load data on demand for responsiveness
- **Memory Management**: Prevent leaks with proper cleanup
- **Optimized Queries**: Efficient data operations
- **Smooth Scrolling**: Virtual scrolling for large lists

### **User Experience**
- **Intuitive Navigation**: Clear workflow and navigation
- **Helpful Feedback**: Status bars, tooltips, and messages
- **Accessibility**: Screen reader support and high contrast
- **Internationalization**: Prepared for translation
- **Professional Polish**: Attention to detail in every interaction

## Advanced Features

### **Data Persistence**
```python
# QSettings for preferences
settings = QSettings("CompanyName", "AppName")
settings.setValue("theme", "dark")
theme = settings.value("theme", "light")

# SQLite for data
import sqlite3
self.conn = sqlite3.connect("app_data.db")
```

### **Custom Painting**
```python
def paintEvent(self, event):
    painter = QPainter(self)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Custom drawing code
```

### **Plugin System**
```python
class PluginInterface:
    """Base class for plugins"""
    def get_name(self):
        raise NotImplementedError
    
    def get_widget(self):
        raise NotImplementedError
```

## Testing Approach

### **Unit Tests**
```python
import unittest
from PyQt6.QtTest import QTest

class TestMainWindow(unittest.TestCase):
    def setUp(self):
        self.app = QApplication([])
        self.window = MainWindow()
    
    def test_button_click(self):
        QTest.mouseClick(self.window.button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.label.text(), "Clicked")
```

## Deployment Considerations

### **Packaging**
- **PyInstaller**: Create standalone executables
- **Requirements**: Clear dependency management
- **Icons**: Application and file type icons
- **Installer**: Platform-specific installers

### **Cross-Platform**
- **Platform Detection**: OS-specific features and paths
- **Native Dialogs**: Use platform file dialogs
- **System Integration**: Tray icons, notifications
- **Testing**: Verify on Windows, macOS, Linux

## Evolution Strategy

### **Iteration Progression**
1. **Foundation (1-3)**: Core functionality with clean UI
2. **Enhancement (4-6)**: Advanced features and polish
3. **Innovation (7+)**: Unique capabilities and integrations

### **Complexity Levels**
- **Basic**: Single window with core features
- **Intermediate**: Multiple windows, threading, persistence
- **Advanced**: Plugins, custom widgets, complex state
- **Expert**: Real-time collaboration, cloud sync, AI features

## Ultra-Thinking Directive

Before creating each PyQt6 application, deeply consider:

**User Needs:**
- What specific problem does this solve?
- Who is the target user and their workflow?
- What would make this indispensable?
- How can we exceed user expectations?

**Technical Excellence:**
- What PyQt6 features best serve this use case?
- How can we ensure responsive performance?
- What patterns make the code maintainable?
- How do we handle edge cases gracefully?

**Visual Design:**
- What styling creates professional appearance?
- How do we ensure visual consistency?
- What animations enhance usability?
- How do we support accessibility?

**Unique Value:**
- What sets this apart from existing solutions?
- What innovative features can we add?
- How do we make it memorable?
- What creates user delight?

**Generate applications that are:**
- **Professionally Polished**: Desktop-quality UI and UX
- **Technically Sound**: Proper PyQt6 patterns and practices
- **Feature Complete**: Solving real problems effectively
- **Immediately Useful**: Ready for actual daily use
- **Delightfully Crafted**: Attention to detail throughout