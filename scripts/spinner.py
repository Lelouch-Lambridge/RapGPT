import sys
import time
import threading
from itertools import cycle

class Spinner:  
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None: cls._instance = super(Spinner, cls).__new__(cls)

    return cls._instance

  def __init__(self, message="Processing..."):
    if hasattr(self, "initialized") and self.initialized: return  

    self.frames = cycle(["⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇"])
    self.running = False
    self.message = message
    self.thread = None
    self.initialized = True

  @classmethod
  def get_instance(cls, message="Processing..."):
    instance = cls()
    instance.message = message
    return instance

  def start(self):
    if self.running: return

    self.running = True
    self.thread = threading.Thread(target=self._spin, daemon=True)
    self.thread.start()

  def _spin(self):
    while self.running:
      sys.stdout.write(f"\r{self.message} {next(self.frames)} ")
      sys.stdout.flush()
      time.sleep(0.1)

  def stop(self):
    self.running = False
    if self.thread: self.thread.join()
    sys.stdout.write("\r" + " " * (len(self.message) + 3) + "\r")
    sys.stdout.flush()
