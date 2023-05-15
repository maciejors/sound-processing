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

    @property
    def fft_magnitude_spectrum_full(self) -> np.ndarray:
        """Returns magnitude spectrum after applying FFT on the whole signal"""
        fft_result = np.fft.fft(self._signal.samples)
        return np.abs(fft_result)

    @property
    def fft_magnitude_spectrum_per_frame(self) -> np.ndarray:
        """Returns magnitude spectrums after applying FFT, per frame"""
        fft_result = np.fft.fft(self._signal.frames, axis=1)
        return np.abs(fft_result)

    @property
    def fft_frequency_values_full(self) -> np.ndarray:
        """Returns the frequency values after applying FFT on the whole signal"""
        freq_vals = np.fft.fftfreq(
            len(self._signal.samples),
            1 / self._signal.sample_rate
        )
        return freq_vals

    @property
    def fft_frequency_values_per_frame(self) -> np.ndarray:
        """Returns frequency values after applying FFT, per frame"""
        freq_vals = [np.fft.fftfreq(len(frame), 1 / self._signal.sample_rate)
                     for frame in self._signal.frames]
        return np.array(freq_vals)

    def volume(self) -> np.ndarray:
        """Returns volume per frame of a signal"""
        return np.mean(self.fft_magnitude_spectrum_per_frame ** 2, axis=1)

    def frequency_cetroids(self) -> np.ndarray:
        """Returns frequency centroids for every frame"""
        return np.sum(self.fft_frequency_values_per_frame * self.fft_magnitude_spectrum_per_frame, axis=1) / \
            np.sum(self.fft_magnitude_spectrum_per_frame, axis=1)

    def effective_bandwidth(self) -> np.ndarray:
        """Returns effective bandwith for every frame"""
        fc = self.frequency_cetroids()
        sq_spectrum = self.fft_magnitude_spectrum_per_frame ** 2
        numerator_inner = (self.fft_magnitude_spectrum_per_frame - fc.reshape(-1, 1)) ** 2 * sq_spectrum
        return np.sqrt(
            np.sum(numerator_inner, axis=1) /
            np.sum(sq_spectrum, axis=1)
        )

    def _band_energy(self, freq_lower_bound: int, freq_upper_bound: int) -> np.ndarray:
        # idx of frequencies in the specified bound
        freqs_idx_masks = [np.where((freqs_frame >= freq_lower_bound) &
                                    (freqs_frame <= freq_upper_bound))
                           for freqs_frame in self.fft_frequency_values_per_frame]
        # magnitudes for frequencies in the bound
        relevant_magnitudes_sq = [frame[mask] ** 2 for frame, mask in
                                  zip(self.fft_magnitude_spectrum_per_frame, freqs_idx_masks)]
        return np.array([np.sum(frame_magn_sq) for frame_magn_sq in relevant_magnitudes_sq])

    def _ersb(self, freq_lower_bound: int, freq_upper_bound: int) -> np.ndarray:
        """Energy Ratio Subband"""
        return self._band_energy(freq_lower_bound, freq_upper_bound) / \
            np.sum(self.fft_magnitude_spectrum_per_frame ** 2, axis=1)

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
        geometric_mean = np.exp(np.mean(np.log(self.fft_magnitude_spectrum_per_frame ** 2), axis=1))
        arithmetic_mean = np.mean(self.fft_magnitude_spectrum_per_frame ** 2, axis=1)
        return geometric_mean / arithmetic_mean

    def spectral_crest_factor(self) -> np.ndarray:
        """Calculates spectral crest factor for each frame"""
        max_squared_magn = np.max(self.fft_magnitude_spectrum_per_frame ** 2, axis=1)
        arithmetic_mean = np.mean(self.fft_magnitude_spectrum_per_frame ** 2, axis=1)
        return max_squared_magn / arithmetic_mean
