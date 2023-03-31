import wave
from functools import cache, cached_property

import numpy as np
import pandas as pd


class WavFile:
    def __init__(self, file, normalise: bool = True):
        with wave.open(file, mode="rb") as wavfile_raw:
            # basic audio properties:
            self.n_channels = wavfile_raw.getnchannels()
            self.n_samples = wavfile_raw.getnframes()
            self.n_samples_per_channel = wavfile_raw.getnframes() // self.n_channels
            self.sample_rate = wavfile_raw.getframerate()
            self.sample_rate_per_channel = wavfile_raw.getframerate() // self.n_channels

            samples_raw = wavfile_raw.readframes(-1)
            samples_all_channels = np.frombuffer(samples_raw, dtype=np.int16)
            # change the type to a larger int (necessary to compute squared values later)
            samples_all_channels = np.array(samples_all_channels, dtype=np.int32)

            # split channels into separate arrays:
            channels = [
                samples_all_channels[i :: self.n_channels]
                for i in range(self.n_channels)
            ]

            # merge them as rows in the final samples array:
            self.samples = np.array(channels)
            # convert to mono by averaging samples from each channel
            self.samples = np.mean(self.samples, axis=0)

            if normalise:
                self.__normalise_samples()

            # TODO - set appropriate frame size and overlap
            self.frames = self.__split_into_frames(20, 0)
            [print(len(f)) for f in self.frames]

    @cached_property
    def audio_length_sec(self) -> float:
        """
        > The function returns the length of the audio in seconds
        :return: The length of the audio in seconds.
        """
        return self.n_samples_per_channel / self.sample_rate_per_channel

    def __normalise_samples(self):
        """
        Normalize the samples to a target level in decibels.
        """
        # samples are averaged so the width at this point is 1
        sample_width = 1

        max_amplitude = np.max(self.samples)
        target_level_db = -3
        target_amplitude = 10 ** (target_level_db / 20) * (
            2 ** (sample_width * 8 - 1) - 1
        )

        gain = target_amplitude / max_amplitude
        samples_normalised = np.floor(self.samples * gain)

        self.samples = samples_normalised

    def __split_into_frames(self, frame_length_ms: int, overlap: int) -> np.ndarray:
        """
        Split the samples array into frames of given size with given overlap.
        :param frame_length_ms: length of the frame in ms
        :param overlap: overlap between frames in samples
        :return: array of frames
        """
        # frame size in samples
        frame_size = frame_length_ms * self.sample_rate_per_channel // 1000
        frames = [self.samples[i: i + frame_size]
                  for i in range(0, self.n_samples_per_channel - frame_size,
                                 frame_size - overlap)]
        return np.array(frames, dtype=object)

    @cached_property
    def volume(self) -> np.ndarray:
        """
        Compute the volume of the audio signal.
        :return: array of volumes of each channel
        """
        return np.sqrt(np.mean(self.frames**2, axis=1))

    @cached_property
    def stereo_balance(self) -> np.ndarray:
        """
        Compute the stereo balance of the audio signal.
        :return: array of stereo balances of each channel
        """
        return self.frames[0] / self.frames[1]

    @cached_property
    def short_time_energy(self) -> np.ndarray:
        """
        Compute the short time energy of the audio signal.
        :return: array of short time energies of each channel
        """
        return np.mean(self.frames**2, axis=1)

    @cached_property
    def zero_crossing_rate(self) -> np.ndarray:
        """
        Compute the zero crossing rate of the audio signal.
        :return: array of zero crossing rates of each channel
        """
        # TODO czy nie trzeba jeszcze dzielić przez 2?
        return np.mean(np.abs(np.diff(np.sign(self.frames))), axis=1)

    @cache
    def get_silence_rate(self, zcr_threshold=0.1, volume_threshold=0.1) -> np.ndarray:
        """Calculates the silent rate of frames based on zero crossing rate and volume.

        Args:
            zcr (numpy.ndarray): Array of zero crossing rate values for each frame.
            volume (numpy.ndarray): Array of volume values for each frame.
            zcr_threshold (float, optional): Zero crossing rate threshold for silence detection. Defaults to 0.1.
            volume_threshold (float, optional): Volume threshold for silence detection. Defaults to 0.1.

        Returns:
            numpy.ndarray: Array of silent rate values for each frame.
        """
        # Compute silent frames based on zero crossing rate and volume thresholds
        silent_frames = np.logical_and(
            self.zero_crossing_rate <= zcr_threshold, self.volume <= volume_threshold
        )

        # Compute silent rate for each frame
        silence_rate = np.mean(silent_frames, axis=-1)

        return silence_rate

    @cached_property
    def fundamental_frequency(self) -> np.ndarray:
        """
        Compute the fundamental frequency of the audio signal.
        :return: array of fundamental frequencies of each channel
        """
        # TODO - implement this
        return np.array([0, 0])

    @cached_property
    def vstd(self) -> np.ndarray:
        """
        Compute the variance of the standard deviation of the audio signal.
        :return: array of variances of standard deviations of each channel
        """
        return np.std(self.frames, axis=1) / np.max(self.frames, axis=1)

    @cached_property
    def volume_dynamic_range(self) -> np.ndarray:
        """
        Compute the volume dynamic range of the audio signal.
        :return: array of volume dynamic ranges of each channel
        """
        return 1 - np.min(self.frames, axis=1) / np.max(self.frames, axis=1)

    @cached_property
    def volume_undulation(self) -> np.ndarray:
        """
        Compute the volume undulation of the audio signal.
        :return: array of volume undulations of each channel
        """
        # TODO - implement this
        return np.std(self.frames, axis=1)

    @cached_property
    def low_short_time_energy_ratio(self) -> np.ndarray:
        """
        Compute the low short time energy of the audio signal.
        :return: array of low short time energies of each channel
        """
        # Compute low energy frames
        low_energy_frames = np.sign(
            0.5 * np.mean(self.short_time_energy) - self.short_time_energy
        )
        lster = np.mean(low_energy_frames + 1) / 2
        return lster

    @cached_property
    def energy_entropy(self) -> np.ndarray:
        """
        Compute the energy entropy of the audio signal.
        :return: array of energy entropies of each channel
        """
        # TODO - implement this
        return np.array([0, 0])

    @cached_property
    def zstd(self) -> np.ndarray:
        """
        Compute the variance of the standard deviation of the zero crossing rate of the audio signal.
        :return: array of variances of standard deviations of zero crossing rates of each channel
        """
        # TODO - implement this
        return np.array([0, 0])

    @cached_property
    def hzcrr(self) -> np.ndarray:
        """
        Compute the high zero crossing rate ratio of the audio signal.
        :return: array of high zero crossing rate ratios of each channel
        """
        # TODO - implement this
        return np.array([0, 0])

    def get_features(self) -> dict:
        """
        Get all features of the audio signal.
        :return: dictionary of features
        """
        return {
            "volume": self.volume,
            "stereo_balance": self.stereo_balance,
            "short_time_energy": self.short_time_energy,
            "zero_crossing_rate": self.zero_crossing_rate,
            "silence_rate": self.get_silence_rate(),
            "fundamental_frequency": self.fundamental_frequency,
            "vstd": self.vstd,
            "volume_dynamic_range": self.volume_dynamic_range,
            "volume_undulation": self.volume_undulation,
            "low_short_time_energy_ratio": self.low_short_time_energy_ratio,
            "energy_entropy": self.energy_entropy,
            "zstd": self.zstd,
            "hzcrr": self.hzcrr,
        }

    def export_features(self, path: str):
        """
        Export all features of the audio signal to a CSV file.
        :param path: path to the CSV file
        """
        features = self.get_features()
        df = pd.DataFrame(features)
        df.to_csv(path, index=False)
