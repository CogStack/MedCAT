import os
from datetime import datetime
from typing import List

from medcat.config import UsageMonitor as UsageMonitorConfig


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

    def log_inference(self,
                      input_text_len: int,
                      trimmed_text_len: int,
                      nr_of_ents_found: int) -> None:
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
