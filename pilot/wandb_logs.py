"""This module defines the `WandbLogs` class.

This class is used to store and retrieve logs from the local file system.
"""

import json
from pathlib import Path


class WandbLogs:
    """A class to handle local logging for wandb."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "wandb_logs.jsonl"

    def save_log(self, data: dict) -> None:
        """Save a log entry to the local log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def get_and_clear_logs(self) -> list[dict]:
        """Get all logs from the local log file and clear it."""
        if not self.log_file.exists():
            return []

        with open(self.log_file, "r+") as f:
            logs = [json.loads(line) for line in f]
            f.truncate(0)
        return logs

    def clear_logs(self) -> None:
        """Clear the local log file."""
        if self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.truncate(0)


