import time

from flwr.common import log as _flwr_log

_start_time: float | None = None


def init_logger(start_time: float):
    """Set the experiment start time for elapsed-second prefixing."""
    global _start_time
    _start_time = start_time


def log(level, msg, *args, **kwargs):
    """Wrapper around flwr.common.log that prepends [<elapsed>s]."""
    if _start_time is not None:
        elapsed = int(time.time() - _start_time)
        msg = f"[{elapsed}s] {msg}"
    _flwr_log(level, msg, *args, **kwargs)
