import os
from datetime import datetime
from typing import List
import platform
import logging

from medcat.config import UsageMonitor as UsageMonitorConfig


LOGS_ENV = "MEDCAT_USAGE_LOGS"
LOGS_LOC_ENV = "MEDCAT_USAGE_LOGS_LOCATION"

DEFAULT_LOGS_WINDOWS = os.path.join(os.environ.get('APPDATA', "NOT WINDOWS"), 'medcat', 'logs')
DEFAULT_LOGS_LINUX = os.path.expanduser("~/.local/share/medcat/logs/")
DEFAULT_LOGS_MACOS = os.path.expanduser("~/Library/Application Support/medcat/logs/")


logger = logging.getLogger(__name__)


class UsageMonitor:

    def __init__(self, model_hash: str, config: UsageMonitorConfig) -> None:
        self.config = config
        self.log_buffer: List[str] = []
        # NOTE: if the model hash changes (i.e model is trained)
        #       then this does not immediately take effect
        self.model_hash = model_hash

    @property
    def log_file(self):
        return os.path.join(
            self.config.log_folder,
            f"{self.config.file_prefix}{self.model_hash}.csv")

    def _get_auto_logs_location(self):
        system = platform.system().lower()
        if system == "windows":
            return DEFAULT_LOGS_WINDOWS
        elif system == "linux":
            return DEFAULT_LOGS_LINUX
        elif system == "darwin":  # macOS
            return DEFAULT_LOGS_MACOS
        else:
            raise OSError(f"Unsupported operating system: {system}")

    def _setup_auto_logs(self):
        # NOTE: os.environ is a snapshot of the environmental variables
        #       from the time that the process was started.
        #       However, someone could still change os.environm manually
        log_dir = os.environ.get(LOGS_LOC_ENV, self._get_auto_logs_location())
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if log_dir != self.config.log_folder:
            self.config.log_folder = log_dir

    def _should_log(self) -> bool:
        if not self.config.enabled:
            logger.warning("Trying to log to file when the usage monitor is "
                           "disabled. This should generally not happen unless "
                           "the config kept track of by the CAT object has is "
                           "different from the one kept track of by the usage "
                           "monitor. So if this keeps coming up, make sure "
                           "everything is up to date")
            return False
        elif self.config.enabled is True:
            return True
        elif self.config.enabled != 'auto':
            raise ValueError("Unknown UsageMonitor enabled status: "
                             f"{self.config.enabled}. Expected one of: "
                             f"True, False, 'auto'")
        # enabled == 'auto'
        env_enabled = os.environ.get(LOGS_ENV, "false").lower()
        if env_enabled in ("false", "0"):
            return False
        self._setup_auto_logs()
        return True

    def log_inference(self,
                      input_text_len: int,
                      trimmed_text_len: int,
                      nr_of_ents_found: int) -> None:
        if not self._should_log():
            return
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp},{input_text_len},{trimmed_text_len},{nr_of_ents_found}"
        self.log_buffer.append(log_entry)
        if len(self.log_buffer) >= self.config.batch_size:
            self._flush_logs()

    def _flush_logs(self) -> None:
        if not self.log_buffer:
            return
        with open(self.log_file, 'a') as f:
            for log_entry in self.log_buffer:
                f.write(log_entry + '\n')
        self.log_buffer = []

    def __del__(self):
        # fail safe for when buffer is non-empty upon application stop (i.e exit call)
        self._flush_logs()
