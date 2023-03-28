import wave

import numpy as np


class WavFile:

    __slots__ = ['samples', 'n_channels', 'n_samples', 'sample_rate', 'sample_width']

    def __init__(self, file, normalise: bool = True):
        with wave.open(file, mode='rb') as wavfile_raw:
            # basic audio properties:
            self.n_channels = wavfile_raw.getnchannels()
            self.n_samples = wavfile_raw.getnframes()
            self.sample_rate = wavfile_raw.getframerate()
            self.sample_width = wavfile_raw.getsampwidth()

            samples_raw = wavfile_raw.readframes(-1)
            samples_all_channels = np.frombuffer(samples_raw, dtype=np.int16)
            # change the type to a larger int (necessary to compute squared values later)
            samples_all_channels = np.array(samples_all_channels, dtype=np.int32)

            # split channels into separate arrays:
            channels = [samples_all_channels[i::self.n_channels]
                        for i in range(self.n_channels)]

            # merge them as rows in the final samples array:
            self.samples = np.array(channels)

            if normalise:
                self.__normalise_samples()

    @property
    def audio_length_sec(self) -> float:
        return self.n_samples / self.sample_rate

    def __normalise_samples(self):
        max_amplitude = np.max(self.samples)
        target_level_db = -3
        target_amplitude = 10 ** (target_level_db / 20) * (2 ** (self.sample_width * 8 - 1) - 1)

        gain = target_amplitude / max_amplitude
        samples_normalised = np.floor(self.samples * gain)

        self.samples = samples_normalised
