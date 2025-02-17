import logging
import os

class Logger:
  _instance = None

  def __new__(cls, log_dir="logs"):
    if cls._instance is None:
      cls._instance = super(Logger, cls).__new__(cls)
      cls._instance._init_logger(log_dir)
    return cls._instance

  def _init_logger(self, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    self.info_logger = logging.getLogger("info_logger")
    self.info_logger.setLevel(logging.INFO)
    info_handler = logging.FileHandler(f"{log_dir}/info.log", mode="a", encoding="utf-8")
    info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    self.info_logger.addHandler(info_handler)

    self.error_logger = logging.getLogger("error_logger")
    self.error_logger.setLevel(logging.WARNING)
    error_handler = logging.FileHandler(f"{log_dir}/error.log", mode="a", encoding="utf-8")
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    self.error_logger.addHandler(error_handler)

  def info(self, message):
    self.info_logger.info(message)

  def warning(self, message):
    self.error_logger.warning(message)

  def error(self, message):
    self.error_logger.error(message)
