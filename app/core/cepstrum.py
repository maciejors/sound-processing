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
        return np.fft.ifft(np.log(np.abs(np.fft.fft(samples))))

    def f0(self) -> np.ndarray:
        def get_pitch_of_frame(frame: np.ndarray) -> float:
            cepstrum = np.real(self.cepstrum(frame))
            cepstrum[0] = 0

            # Frequency bounds
            period_ub = 1 / 50  # Highest frequency (50 Hz)
            period_lb = 1 / 400  # Lowest frequency (400 Hz)

            # Find local maxima within the frequency bounds
            cepstrum_freqs = np.fft.fftfreq(len(cepstrum), 1 / self._signal.sample_rate)
            mask = (cepstrum_freqs >= period_lb) & (cepstrum_freqs <= period_ub)
            local_maxima_indices = np.where((cepstrum == np.maximum(cepstrum, np.roll(cepstrum, 1))) &
                                            (cepstrum == np.maximum(cepstrum, np.roll(cepstrum, -1))) &
                                            mask)[0]

            if len(local_maxima_indices) > 0:
                # Get the index of the highest local maximum
                argmax = np.argmax(cepstrum[local_maxima_indices])
                index = local_maxima_indices[argmax]
                return self._signal.sample_rate / cepstrum_freqs[index] / 2
            else:
                return 0.0

        pitches = [get_pitch_of_frame(f) for f in self._signal.frames]
        return np.array(pitches)
