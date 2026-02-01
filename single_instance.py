"""
Single Instance Application Support

Ensures only one instance of the application runs at a time.
When a second instance is launched with file arguments, it sends
the files to the existing instance and exits.
"""

import sys
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket


class SingleInstance(QObject):
    """Manage single instance application using local sockets"""
    
    files_received = pyqtSignal(list)  # Emitted when primary receives files from secondary
    
    def __init__(self, app_id='ImageViewer', parent=None):
        super().__init__(parent)
        self.app_id = app_id
        self.server = None
        self.is_primary = False
        
    def check_and_connect(self):
        """
        Check if another instance is running.
        Returns True if this is the primary instance, False if secondary.
        """
        # Try to connect to an existing instance
        socket = QLocalSocket()
        socket.connectToServer(self.app_id)
        
        # If connection succeeds, another instance is running
        if socket.waitForConnected(500):
            socket.disconnectFromServer()
            self.is_primary = False
            return False
        
        # No existing instance, so we're the primary
        # Remove any stale server socket
        QLocalServer.removeServer(self.app_id)
        
        # Create our server
        self.server = QLocalServer(self)
        self.server.newConnection.connect(self._on_new_connection)
        
        if not self.server.listen(self.app_id):
            # Another process might have won the race to listen.
            # Try connecting one last time.
            socket = QLocalSocket()
            socket.connectToServer(self.app_id)
            if socket.waitForConnected(500):
                socket.disconnectFromServer()
                self.is_primary = False
                return False
            
            # If even connection fails, we are in a weird state (e.g. permission error)
            # Proceed as primary for usability, but it's a fallback.
            print(f"Warning: Could not start single instance server or connect: {self.server.errorString()}")
            self.is_primary = True
            return True
        
        self.is_primary = True
        return True
    
    def send_files(self, file_paths):
        """
        Send file paths to the primary instance.
        Returns True if successful, False otherwise.
        """
        if not file_paths:
            return True
        
        # Filter out command-line flags
        files = [f for f in file_paths if not f.startswith('-')]
        if not files:
            return True
        
        socket = QLocalSocket()
        socket.connectToServer(self.app_id)
        
        if not socket.waitForConnected(1000):
            print(f"Error: Could not connect to primary instance: {socket.errorString()}")
            return False
        
        # Send file paths as newline-separated string
        message = '\n'.join(files)
        socket.write(message.encode('utf-8'))
        socket.flush()
        
        if not socket.waitForBytesWritten(1000):
            print(f"Error: Could not send files to primary instance")
            return False
        
        socket.disconnectFromServer()
        return True
    
    def _on_new_connection(self):
        """Handle new connection from secondary instance"""
        if not self.server:
            return
        
        client = self.server.nextPendingConnection()
        if not client:
            return
        
        # Wait for data
        if client.waitForReadyRead(1000):
            data = client.readAll().data().decode('utf-8')
            files = [f.strip() for f in data.split('\n') if f.strip()]
            
            if files:
                self.files_received.emit(files)
        
        client.disconnectFromServer()
        client.deleteLater()
