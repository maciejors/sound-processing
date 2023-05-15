import numpy as np

from app.models.signal import Signal


class FrequencyAnalyser:
    """
    Class that provides methods to calculate various frequency-based
    parameters of a signal. It utilises Fast Fourier Transformation.
    """

    __slots__ = ['_signal']

    def __init__(self, signal: Signal):
        self._signal = signal

    def volume(self) -> np.ndarray:
        """Returns volume per frame of a signal"""
        return np.mean(self._signal.fft_magn_spectr_frames ** 2, axis=1)

    def frequency_cetroids(self) -> np.ndarray:
        """Returns frequency centroids for every frame"""
        numerator = np.sum(
            self._signal.fft_freq_vals_frames * self._signal.fft_magn_spectr_frames,
            axis=1
        )
        denom = np.sum(self._signal.fft_magn_spectr_frames, axis=1)
        return numerator / denom

    def effective_bandwidth(self) -> np.ndarray:
        """Returns effective bandwith for every frame"""
        fc = self.frequency_cetroids()
        sq_spectrum = self._signal.fft_magn_spectr_frames ** 2
        numerator_inner = \
            (self._signal.fft_magn_spectr_frames - fc.reshape(-1, 1)) ** 2 * sq_spectrum
        return np.sqrt(
            np.sum(numerator_inner, axis=1) /
            np.sum(sq_spectrum, axis=1)
        )

    def _band_energy(self, freq_lower_bound: int, freq_upper_bound: int) -> np.ndarray:
        # idx of frequencies in the specified bound
        freqs_idx_masks = [np.where((freqs_frame >= freq_lower_bound) &
                                    (freqs_frame <= freq_upper_bound))
                           for freqs_frame in self._signal.fft_freq_vals_frames]
        # magnitudes for frequencies in the bound
        relevant_magnitudes_sq = [frame[mask] ** 2 for frame, mask in
                                  zip(self._signal.fft_magn_spectr_frames, freqs_idx_masks)]
        return np.array([np.sum(frame_magn_sq) for frame_magn_sq in relevant_magnitudes_sq])

    def _ersb(self, freq_lower_bound: int, freq_upper_bound: int) -> np.ndarray:
        """Energy Ratio Subband"""
        return self._band_energy(freq_lower_bound, freq_upper_bound) / \
            np.sum(self._signal.fft_magn_spectr_frames ** 2, axis=1)

    def ersb1(self) -> np.ndarray:
        """Energy Ratio Subband for frequencies ranging from 0 to 630 Hz"""
        return self._ersb(0, 630)

    def ersb2(self) -> np.ndarray:
        """Energy Ratio Subband for frequencies ranging from 630 to 1720 Hz"""
        return self._ersb(630, 1720)

    def ersb3(self) -> np.ndarray:
        """Energy Ratio Subband for frequencies ranging from 1720 to 4400 Hz"""
        return self._ersb(1720, 4400)

    def spectral_flatness(self) -> np.ndarray:
        """Calculates spectral flatness for each frame"""
        geometric_mean = np.exp(np.mean(np.log(
            self._signal.fft_magn_spectr_frames ** 2), axis=1))
        arithmetic_mean = np.mean(self._signal.fft_magn_spectr_frames ** 2, axis=1)
        return geometric_mean / arithmetic_mean

    def spectral_crest_factor(self) -> np.ndarray:
        """Calculates spectral crest factor for each frame"""
        max_squared_magn = np.max(self._signal.fft_magn_spectr_frames ** 2, axis=1)
        arithmetic_mean = np.mean(self._signal.fft_magn_spectr_frames ** 2, axis=1)
        return max_squared_magn / arithmetic_mean
