import numpy as np
import math
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
        return np.fft.ifft(np.log(np.abs(np.fft.fft(samples))))
    
    def f0(self) -> np.ndarray:
        sig_to_frames = self._signal.frames
        frame_len,frame_num = self._signal.frame_size,self._signal.n_frames
        peaks = []
        for i in range(frame_num):
            frame = sig_to_frames[:,i]
            freq = np.fft.fftfreq(frame_len,1/self._signal.sample_rate)
            m = self._signal.fft_magn_spectr_frames[:,i]
            for j in range(m.shape[0]):
                if freq[j] < 50 or freq[j] > 400:
                    m[j] = 0
            mid = math.ceil(frame_len//2)
            magn_max_id = np.argmax(m[:mid])
            peaks.append(freq[magn_max_id])
        return peaks
