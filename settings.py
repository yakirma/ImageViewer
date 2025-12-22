from PyQt6.QtCore import QSettings

MAX_RECENT_FILES = 10
SETTINGS_RECENT_FILES_KEY = "recentFiles"
MAX_RAW_HISTORY = 100
SETTINGS_RAW_HISTORY_KEY = "rawFileHistory"


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

def load_raw_history():
    settings = QSettings("ImageViewer", "ImageViewer")
    return settings.value(SETTINGS_RAW_HISTORY_KEY, {}, type=dict)

def save_raw_history(history):
    settings = QSettings("ImageViewer", "ImageViewer")
    settings.setValue(SETTINGS_RAW_HISTORY_KEY, history)

def update_raw_history(file_path, params):
    """
    Updates the history with params for file_path.
    params should be a dict: {'width': w, 'height': h, 'dtype': d, 'color_format': f}
    """
    history = load_raw_history()
    
    # Remove if exists to re-insert at end (LRU behavior relies on dict order in recent Python, 
    # but for safety/clarity we just re-insert).
    # Ideally for LRU we want the most recent at the END or START. 
    # Let's say we keep most recent at the END (insertion order).
    if file_path in history:
        del history[file_path]
        
    history[file_path] = params
    
    # Prune
    while len(history) > MAX_RAW_HISTORY:
        # Remove first key (oldest)
        # iter(history) gives keys in insertion order (Py 3.7+)
        oldest_key = next(iter(history))
        del history[oldest_key]
        
    save_raw_history(history)