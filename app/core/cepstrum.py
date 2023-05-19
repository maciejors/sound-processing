import numpy as np

from app.models.signal import Signal


class Cepstrum:
    """
    Class that provides methods related to Cepstrum.
    """

    __slots__ = ['_signal']

    def __init__(self, signal: Signal):
        self._signal = signal

    @staticmethod
    def cepstrum(samples: np.ndarray) -> np.ndarray:
        return np.fft.fft(np.log(np.abs(np.fft.fft(samples))))

    def f0(self) -> np.ndarray:
        def get_pitch_of_frame(frame: np.ndarray) -> float:
            cepstrum = self.cepstrum(frame)
            cepstrum[0] = 0
            # first max
            argmax = np.argmax(np.abs(cepstrum))
            return self._signal.sample_rate / (2 * argmax)

        pitches = [get_pitch_of_frame(f) for f in self._signal.frames]
        return np.array(pitches)
