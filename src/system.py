import os
import psutil
from .utils import DOWNLOADS_DIR, TEMP_DATA_DIR

def get_system_resources():
    """Get system memory and disk usage - Application usage vs Available"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Get Spark process memory
    process = psutil.Process(os.getpid())
    app_mem = process.memory_info().rss

    # Calculate app disk usage (downloads + temp data)
    app_disk_usage = 0
    for folder in [DOWNLOADS_DIR, TEMP_DATA_DIR]:
        if os.path.exists(folder):
            for dirpath, dirnames, filenames in os.walk(folder):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        app_disk_usage += os.path.getsize(filepath)
                    except:
                        pass

    return {
        'app_mem_gb': app_mem / (1024**3),
        'mem_available_gb': mem.available / (1024**3),
        'mem_total_gb': mem.total / (1024**3),
        'mem_percent': (app_mem / mem.available) * 100 if mem.available > 0 else 0,
        'app_disk_gb': app_disk_usage / (1024**3),
        'disk_available_gb': disk.free / (1024**3),
        'disk_total_gb': disk.total / (1024**3),
        'disk_percent': (app_disk_usage / disk.free) * 100 if disk.free > 0 else 0,
    }
