import json
import time
import numpy as np
from datetime import datetime
class CustomEncoder(json.JSONEncoder):
    """
    Safely handles Numpy data types and custom Python objects so they don't crash the JSON logger.
    """
    def default(self, obj):
        # 1. Safely convert Numpy integers (This fixes the int64 crash!)
        if isinstance(obj, np.integer):
            return int(obj)
        
        # 2. Safely convert Numpy floats
        if isinstance(obj, np.floating):
            return float(obj)
        
        # 3. Safely convert Numpy arrays to standard Python lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # 4. Safely convert custom objects (Person, PoseData, HandData) to dictionaries
        if hasattr(obj, '__dict__'):
            return obj.__dict__
            
        # Fallback to the default JSON encoder if it's a standard type
        return super().default(obj)


class JsonlLogger:
    def __init__(self, log_path="emmaeye_log.jsonl"):
        self.log_path = log_path
        # Open in append mode ('a') so we don't overwrite previous sessions
        self.file = open(self.log_path, 'a', encoding='utf-8')

    def log_people(self, people):
        """
        Logs the current frame's data as a single JSON line.
        """
        if not people:
            return

        import time
        from datetime import datetime
        
        # Calculate standard UNIX epoch time
        unix_ms = int(time.time() * 1000)
        
        # Calculate milliseconds after midnight
        now = datetime.now()
        ms_after_midnight = int((now.hour * 3600 + now.minute * 60 + now.second) * 1000 + now.microsecond / 1000)

        frame_data = {
            "timestamp_ms": unix_ms,             # Keep the original UNIX timestamp
            "ms_after_midnight": ms_after_midnight, # Add the new simulator timestamp
            "people": people
        }

        try:
            # The `cls=CustomEncoder` is what intercepts and fixes the int64s!
            json_line = json.dumps(frame_data, cls=CustomEncoder)
            self.file.write(json_line + '\n')
            
            # Force write to disk immediately so no data is lost if the script closes
            self.file.flush() 
        except Exception as e:
            print(f"Logger Error: {e}")

    def close(self):
        """
        Safely close the file stream when the program exits.
        """
        if self.file and not self.file.closed:
            self.file.close()