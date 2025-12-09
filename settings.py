from PyQt6.QtCore import QSettings

MAX_RECENT_FILES = 10
SETTINGS_RECENT_FILES_KEY = "recentFiles"


def load_recent_files():
    settings = QSettings("ImageViewer", "ImageViewer")
    return settings.value(SETTINGS_RECENT_FILES_KEY, [], type=list)


def save_recent_files(files):
    settings = QSettings("ImageViewer", "ImageViewer")
    settings.setValue(SETTINGS_RECENT_FILES_KEY, files)


def add_to_recent_files(files, file_path):
    if file_path in files:
        files.remove(file_path)
    files.insert(0, file_path)
    while len(files) > MAX_RECENT_FILES:
        files.pop()
    save_recent_files(files)
    return files