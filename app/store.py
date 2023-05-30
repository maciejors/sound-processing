from typing import Optional
from app.models.signal import Signal

__signal: Optional[Signal] = None


def get_signal() -> Optional[Signal]:
    return __signal


def update_signal(raw_file):
    global __signal
    __signal = Signal(raw_file)
