import os
import threading
import queue
import time
from openal import oalOpen, oalQuit, AL_PLAYING

NUMBER_OF_VARIATIONS = 1  # Number of variations per sign

class GestureAudioManager:

    def __init__(self):
        self.sources = {}
        self.queues = {}
        self.audio_files = {} # Pre-load paths here
        
        # Start a background monitor thread to process queues
        self.running = True
        self.monitor_thread = threading.Thread(target=self._process_queues, daemon=True)
        self.monitor_thread.start()

        self.warning_source = None
        self.is_warning_active = False

        # Pre-load audio buffers into memory
        self.audio_buffers = {}
        self._load_all_audio()

    def _load_all_audio(self):
      """Call this once at start to load all files into memory."""
      for file in os.listdir("FabianAndFerProject/audio"):
          if file.endswith(".wav"):
              name = os.path.splitext(file)[0]
              # Store the buffer itself, not the path
              self.audio_buffers[name] = oalOpen(f"FabianAndFerProject/audio/{file}").buffer

      # Initialize the dedicated warning source if the file exists
      if "beep_warning" in self.audio_buffers:
        # We open an empty source and assign the buffer
        self.warning_source = oalOpen("FabianAndFerProject/audio/beep_warning.wav")
        self.warning_source.set_looping(True) # type: ignore
      else:
        print("Warning: beep_warning.wav not found in audio folder.")
    
    def toggle_warning(self, state=None):
      """
      Toggles the warning beep on/off. 
      If state is provided (True/False), it sets it explicitly.
      """
      if self.warning_source is None:
        return

      # If no explicit state is provided, flip the current state
      if state is None:
        self.is_warning_active = not self.is_warning_active
      else:
        self.is_warning_active = state

      if self.is_warning_active:
        if self.warning_source.get_state() != AL_PLAYING:
          self.warning_source.play()
      else:
        self.warning_source.stop()

    def trigger_gesture(self, person_id, position, sign_name):
        """Adds a gesture to the queue for a specific person."""
        if sign_name == "UNKNOWN":
            return  # Don't trigger sound for unknown gestures
        if person_id not in self.queues:
            self.queues[person_id] = queue.Queue()
            self.sources[person_id] = oalOpen(f"FabianAndFerProject/audio/{sign_name}_{(person_id % NUMBER_OF_VARIATIONS)+1}.wav") 
        
        # Queue the gesture (store position and sign)
        self.queues[person_id].put((position, sign_name))

    def _process_queues(self):
        """Background thread that manages playing queued sounds."""
        while self.running:
            for person_id, q in self.queues.items():
                source = self.sources[person_id]
                
                # If source is free and there is something in the queue
                if source.get_state() != AL_PLAYING and not q.empty():
                    position, sign_name = q.get()

                    if sign_name == "UNKNOWN":
                        continue  # Don't trigger sound for unknown gestures
                    
                    buffer_key = f"{sign_name}_{(person_id % NUMBER_OF_VARIATIONS)+1}"
                    print(f"Switching {person_id} to {buffer_key}") # <--- Add this!

                    if buffer_key in self.audio_buffers:
                        source.set_position(position)
                        source._set_buffer(self.audio_buffers[buffer_key])
                        source.play()
                    else:
                        print(f"Error: {buffer_key} not found in buffers!")

                    # # Update position and play
                    # source.set_position(position)

                    # source.buffer = self.audio_buffers[f"{sign_name}_{(person_id % NUMBER_OF_VARIATIONS)+1}"] # Use pre-loaded buffer

                    # source.play()
            
            time.sleep(0.1) # Prevent CPU spiking

    def cleanup(self):
        self.running = False
        if self.warning_source:
            self.warning_source.stop()
        oalQuit()
