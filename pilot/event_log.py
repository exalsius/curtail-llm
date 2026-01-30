import csv
import os
import threading
import time


class EventLog:
    """Thread-safe CSV event logger writing to logs/events.csv."""

    _COLUMNS = ["wall_time", "elapsed_s", "event_type", "round", "client", "details"]

    def __init__(self, path: str = "logs/events.csv"):
        self._path = path
        self._lock = threading.Lock()
        self._start_time = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self._path, "w", newline="") as f:
            csv.writer(f).writerow(self._COLUMNS)

    def log(
        self,
        event_type: str,
        round: int | str = "",
        client: str = "",
        details: str = "",
    ):
        wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed_s = f"{time.time() - self._start_time:.1f}"
        row = [wall_time, elapsed_s, event_type, round, client, details]
        with self._lock:
            with open(self._path, "a", newline="") as f:
                csv.writer(f).writerow(row)


# Module-level singleton, initialised by server_app.py
_event_log: EventLog | None = None


def init_event_log(path: str = "logs/events.csv") -> EventLog:
    global _event_log
    _event_log = EventLog(path)
    return _event_log


def get_event_log() -> EventLog | None:
    return _event_log
