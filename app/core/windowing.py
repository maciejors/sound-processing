from typing import Callable

import numpy as np
import scipy.signal.windows as wn

from app.models.signal import Signal


class Windowing:
    """
    Class that provides methods to handle signal windowing.
    """

    __slots__ = ['_signal', 'windowing_func']

    def __init__(self, signal: Signal):
        self._signal = signal
        # default window func
        self.windowing_func = wn.hamming

    @property
    def function_map(self) -> dict[str, Callable]:
        """Returns a dict that maps user-friendly windowing function names to
        actual function objects"""
        return {
            'Rectangular': wn.boxcar,
            'Triangular': wn.triang,
            'Bartlett': wn.bartlett,
            'Blackman': wn.blackman,
            'van Hann': wn.hann,
            'Hamming': wn.hamming,
            'Parzen': wn.parzen,
            'Taylor': wn.taylor,
        }

    def set_windowing_func(self, func_name: str):
        try:
            self.windowing_func = self.function_map[func_name]
        except KeyError:
            raise ValueError('No windowing function with that name available. '
                             f'Available functions: {self.function_map.keys()}')

    def windowed_samples(self) -> np.ndarray:
        """Applies windowing to original samples"""
        window = self.windowing_func(self._signal.n_samples)
        return self._signal.samples * window

    def windowed_fft_freqs(self) -> np.ndarray:
        """Returns x-axis of frequency domain plot for a signal with windowing applied"""
        freq_vals = np.fft.fftfreq(
            len(self.windowed_samples()),
            1 / self._signal.sample_rate
        )
        return freq_vals

    def windowed_fft_magn_spectr(self) -> np.ndarray:
        """Returns fft magnitude spectrum for a signal with windowing applied"""
        fft_result = np.fft.fft(self.windowed_samples())
        return np.abs(fft_result)

    def spectrogram(self) -> np.ndarray:
        spectrogram = np.zeros((self._signal.frame_size // 2 + 1, self._signal.n_frames))
        # STFT
        for i, frame in enumerate(self._signal.frames):
            spectrum = np.fft.fft(frame * self.windowing_func(len(frame)))
            magnitude_spectrum = np.abs(spectrum[:self._signal.frame_size // 2 + 1])
            spectrogram[:, i] = magnitude_spectrum

        return 20 * np.log10(spectrogram)
