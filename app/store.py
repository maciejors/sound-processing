from typing import Optional
from app.models.wavfile import WavFile

__wavfile: Optional[WavFile] = None


def get_wavfile() -> Optional[WavFile]:
    return __wavfile


def update_wavfile(raw_file):
    global __wavfile
    __wavfile = WavFile(raw_file)
