"""
Single Instance Application Support

Ensures only one instance of the application runs at a time.
When a second instance is launched with file arguments, it sends
the files to the existing instance and exits.
"""

import json
import os
import struct
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

_HEADER_FMT = '>I'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 64 * 1024 * 1024
_SEND_TIMEOUT_S = 5.0


class SingleInstance(QObject):
    """Manage single instance application using local sockets"""
    
    files_received = pyqtSignal(list)  # Emitted when primary receives files from secondary
    
    def __init__(self, app_id='ImageViewer', parent=None):
        super().__init__(parent)
        self.app_id = app_id
        self.server = None
        self.is_primary = False
        self._buffers = {}  # QLocalSocket -> bytearray of received bytes

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

        Paths are made absolute here, because the primary resolves them
        against its own working directory (often '/' when launched from
        the Finder), not ours.

        Wire format: a 4-byte big-endian length followed by that many bytes
        of UTF-8 JSON (a list of path strings). The receiver accumulates
        bytes asynchronously until the whole message has arrived, so long
        file lists split across several socket chunks are not truncated.

        Returns True if successful, False otherwise.
        """
        files = absolute_paths(file_paths)
        if not files:
            return True

        socket = QLocalSocket()
        socket.connectToServer(self.app_id)

        if not socket.waitForConnected(1000):
            print(f"Error: Could not connect to primary instance: {socket.errorString()}")
            return False

        socket.write(encode_message(files))

        # Large payloads may need several writes; keep pumping until the
        # whole message has been handed to the OS or we time out.
        deadline = time.monotonic() + _SEND_TIMEOUT_S
        while socket.bytesToWrite() > 0:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                print("Error: Could not send files to primary instance")
                socket.abort()
                return False
            socket.waitForBytesWritten(min(remaining_ms, 1000))

        socket.disconnectFromServer()
        if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            socket.waitForDisconnected(1000)
        return True

    def _on_new_connection(self):
        """Handle new connections from secondary instances without blocking.

        Bytes are accumulated per socket via readyRead; the message is
        parsed once the length-prefixed payload is complete.
        """
        if not self.server:
            return

        while True:
            client = self.server.nextPendingConnection()
            if not client:
                return
            self._buffers[client] = bytearray()
            client.readyRead.connect(lambda c=client: self._on_ready_read(c))
            client.disconnected.connect(lambda c=client: self._on_client_disconnected(c))
            # Data may already be buffered before the signals were connected.
            if client.bytesAvailable() > 0:
                self._on_ready_read(client)

    def _on_ready_read(self, client):
        buf = self._buffers.get(client)
        if buf is None:
            return
        buf.extend(client.readAll().data())

        if len(buf) >= _HEADER_SIZE:
            (length,) = struct.unpack(_HEADER_FMT, bytes(buf[:_HEADER_SIZE]))
            if length > _MAX_MESSAGE_SIZE:
                print("Warning: Rejecting oversized single-instance message")
                self._finish_client(client)
                return
            if len(buf) >= _HEADER_SIZE + length:
                payload = bytes(buf[_HEADER_SIZE:_HEADER_SIZE + length])
                self._finish_client(client)
                files = decode_payload(payload)
                if files:
                    self.files_received.emit(files)

    def _on_client_disconnected(self, client):
        # Drain anything that arrived together with the disconnect.
        if client in self._buffers and client.bytesAvailable() > 0:
            self._on_ready_read(client)
        if client in self._buffers:
            # Incomplete message: drop it.
            self._finish_client(client)

    def _finish_client(self, client):
        if self._buffers.pop(client, None) is None:
            return
        try:
            client.readyRead.disconnect()
            client.disconnected.disconnect()
        except TypeError:
            pass
        client.disconnectFromServer()
        client.deleteLater()


def absolute_paths(args):
    """Drop command-line flags and make the remaining paths absolute."""
    return [os.path.abspath(a) for a in args if a and not a.startswith('-')]


def encode_message(files):
    payload = json.dumps(list(files)).encode('utf-8')
    return struct.pack(_HEADER_FMT, len(payload)) + payload


def decode_payload(payload):
    try:
        data = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        print("Warning: Ignoring malformed single-instance message")
        return []
    if not isinstance(data, list):
        return []
    return [f for f in data if isinstance(f, str) and f]
