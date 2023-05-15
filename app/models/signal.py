import wave
from functools import cache, cached_property

import numpy as np


class Signal:
    def __init__(self, file, normalise: bool = True,
                 frame_length_ms: int = 20, frame_overlap: int = 0):
        """
        :param file: a wavfile (path or bytes)
        :param normalise: whether or not to normalise the audio
        :param frame_length_ms: length of the frame in ms
        :param frame_overlap: overlap between frames in samples
        """
        self.frame_length_ms = frame_length_ms
        self.frame_overlap = frame_overlap

        with wave.open(file, mode="rb") as wavfile_raw:
            # basic audio properties:
            self.n_channels = wavfile_raw.getnchannels()
            self.n_samples_all = wavfile_raw.getnframes()
            self.sample_rate = wavfile_raw.getframerate()

            samples_raw = wavfile_raw.readframes(-1)
            samples_all_channels = np.frombuffer(samples_raw, dtype=np.int16)
            # change the type to a larger int (necessary to compute squared values later)
            samples_all_channels = np.array(samples_all_channels, dtype=np.int32)

            # split channels into separate arrays:
            channels = [
                samples_all_channels[i:: self.n_channels]
                for i in range(self.n_channels)
            ]

            # merge them as rows in the final samples array:
            self.samples_all = np.array(channels)
            # convert to mono by averaging samples from each channel
            self.samples_all = np.mean(self.samples_all, axis=0)

            if normalise:
                self.__normalise_samples()

            # set the default boundaries of audio to analyse
            self.__start_id = 0
            self.__end_id = len(self.samples_all) - 1

            self.frames = split_samples_into_frames(
                self.samples, self.frame_size, self.frame_overlap)

    @property
    def boundaries(self) -> tuple[int, int]:
        return self.__start_id, self.__end_id

    @property
    def samples(self) -> np.ndarray:
        return self.samples_all[self.__start_id:self.__end_id]

    @property
    def n_samples(self) -> int:
        return self.__end_id - self.__start_id

    @property
    def n_samples_in_frames(self) -> int:
        return sum([len(f) for f in self.frames])

    @property
    def all_audio_length_sec(self) -> float:
        """
        > The function returns the length of the audio in seconds (it ignores boundaries)
        :return: The length of the audio in seconds.
        """
        return self.n_samples_all / self.sample_rate

    @property
    def audio_length_sec(self) -> float:
        """
        > The function returns the length of the audio in seconds (it considers boundaries)
        :return: The length of the audio in seconds.
        """
        return self.n_samples / self.sample_rate

    def __normalise_samples(self):
        """
        Normalize the samples to a target level in decibels.
        """
        # samples are averaged so the width at this point is 1
        sample_width = 1

        max_amplitude = np.max(self.samples_all)
        target_level_db = -3
        target_amplitude = 10 ** (target_level_db / 20) * (
                2 ** (sample_width * 8 - 1) - 1
        )

        gain = target_amplitude / max_amplitude
        samples_normalised = np.floor(self.samples_all * gain)

        self.samples_all = samples_normalised

    @property
    def frame_size(self) -> int:
        """Number of samples in a single frame"""
        return self.frame_length_ms * self.sample_rate // 1000

    def set_boundaries(self, start_sample_id: int, end_sample_id: int):
        """
        Sets the starting and the ending point of audio to analyse.
        """
        self.__start_id = start_sample_id
        self.__end_id = end_sample_id
        self.frames = split_samples_into_frames(
            self.samples, self.frame_size, self.frame_overlap)

    @cached_property
    def n_frames(self) -> int:
        return len(self.frames)

    def get_features(self) -> dict:
        """
        Get all features of the audio signal.
        :return: dictionary of features
        """
        return {
            "frame-level": {
                "volume": list(self.volume),
                "short_time_energy": list(self.short_time_energy),
                "zero_crossing_rate": list(self.zero_crossing_rate),
                "silence_rate": self.get_silence_rate(),
                "fundamental_frequency": list(self.fundamental_frequency),
            },
            "clip-level": {
                "vstd": self.vstd,
                "volume_dynamic_range": self.volume_dynamic_range,
                "volume_undulation": self.volume_undulation,
                "low_short_time_energy_ratio": self.low_short_time_energy_ratio,
                "energy_entropy": self.energy_entropy(),
                "zstd": self.zstd,
                "hzcrr": self.hzcrr,
            },
        }

    # ========== FRAME-LEVEL ========== #

    @cached_property
    def volume(self) -> np.ndarray:
        """
        Compute the volume of the audio signal.
        :return: array of volumes of each channel
        """
        return np.sqrt(np.mean(self.frames ** 2, axis=1))

    @cached_property
    def short_time_energy(self) -> np.ndarray:
        """
        Compute the short time energy of the audio signal.
        :return: array of short time energies of each channel
        """
        return np.mean(self.frames ** 2, axis=1)

    @cached_property
    def zero_crossing_rate(self) -> np.ndarray:
        """
        Compute the zero crossing rate of the audio signal.
        :return: array of zero crossing rates of each channel
        """
        return np.mean(np.abs(np.diff(np.sign(self.frames))), axis=1) / 2

    @cache
    def get_silent_frames(self, zcr_threshold=0.1, volume_threshold=0.1) -> np.ndarray:
        """Calculates the silent frames based on zero crossing rate and volume.

        Args:
            zcr (numpy.ndarray): Array of zero crossing rate values for each frame.
            volume (numpy.ndarray): Array of volume values for each frame.
            zcr_threshold (float, optional): Zero crossing rate threshold for silence detection. Defaults to 0.1.
            volume_threshold (float, optional): Volume threshold for silence detection. Defaults to 0.1.

        Returns:
            numpy.ndarray: Array of silent frames.
        """
        # Compute silent frames based on zero crossing rate and volume thresholds
        silent_frames = np.logical_and(
            self.zero_crossing_rate <= zcr_threshold, self.volume <= volume_threshold
        ).astype(int)
        return silent_frames

    @cache
    def get_frame_types(self, zcr_threshold=0.1, volume_threshold=0.1) -> np.ndarray:
        """Calculates the frame types based on zero crossing rate and volume.
            0 - silent
            1 - voiceless
            2 - voiced

        Args:
            zcr (numpy.ndarray): Array of zero crossing rate values for each frame.
            volume (numpy.ndarray): Array of volume values for each frame.
            zcr_threshold (float, optional): Zero crossing rate threshold for silence detection. Defaults to 0.1.
            volume_threshold (float, optional): Volume threshold for silence detection. Defaults to 0.1.

        Returns:
            numpy.ndarray: Array of frame types.
        """
        # Compute frame types based on zero crossing rate and volume thresholds
        frame_types = np.where(
            self.zero_crossing_rate <= zcr_threshold,
            np.where(self.volume <= volume_threshold, 0, 2), 1
        )
        # TODO - possible situation when frame has high zcr and low volume
        return frame_types

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
        silent_frames = self.get_silent_frames(zcr_threshold, volume_threshold)

        # Compute silent rate for each frame
        silence_rate = np.mean(silent_frames, axis=-1)

        return silence_rate

    @cached_property
    def fundamental_frequency(self) -> np.ndarray:
        """
        Compute the fundamental frequency of the audio signal.
        :return: array of fundamental frequencies of each channel
        """
        freqs = []
        for frame in self.frames:
            # Compute autocorrelation of the frame
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(frame) - 1:]
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if not len(zero_crossings):
                freqs.append(0)
                continue
            autocorr = autocorr[zero_crossings[0]:]

            # Find the first peak in the autocorrelation
            peak = np.argmax(autocorr) + zero_crossings[0]

            # Convert lag to time and frequency
            lag = peak / self.sample_rate
            freq = 1 / lag

            freqs.append(freq)

        return np.array(freqs)

    # ========== CLIP-LEVEL ========== #

    @cached_property
    def vstd(self) -> float:
        """
        Compute the variance of the standard deviation of the audio signal.
        :return: array of variances of standard deviations of each channel
        """
        return np.std(self.samples) / np.max(self.samples)

    @cached_property
    def volume_dynamic_range(self) -> float:
        """
        Compute the volume dynamic range of the audio signal.
        :return: array of volume dynamic ranges of each channel
        """
        return 1 - np.min(self.samples) / np.max(self.samples)

    @cached_property
    def volume_undulation(self) -> float:
        # """
        # Compute the volume undulation of the audio signal.
        # :return: array of volume undulations of each channel
        # """
        # # Compute RMS amplitude of audio signal
        # rms_amplitude = np.sqrt(np.mean(np.square(self.frames), axis=1))

        # # Compute time array in seconds
        # time = np.arange(0, len(audio_signal)) / sample_rate

        # # Compute RMS amplitude over time
        # rms_amplitude_over_time = np.sqrt(np.mean(np.square(audio_signal), axis=1))

        # # Compute differences between adjacent RMS amplitude values
        # differences = np.diff(rms_amplitude_over_time)

        # # Compute mean of absolute differences
        # volume_undulation = np.mean(np.abs(differences))

        # return volume_undulation
        # Initialize the VU value to zero.
        VU = 0.0

        # Compute the differences between neighboring peaks and valleys.
        peaks = np.logical_and(self.volume[1:-1] > self.volume[:-2],
                               self.volume[1:-1] > self.volume[2:])
        valleys = np.logical_and(self.volume[1:-1] < self.volume[:-2],
                                 self.volume[1:-1] < self.volume[2:])
        peak_diffs = self.volume[1:-1] - np.minimum(self.volume[:-2], self.volume[2:])
        valley_diffs = np.maximum(self.volume[:-2], self.volume[2:]) - self.volume[1:-1]

        # Compute the VU value by summing the differences between neighboring peaks and valleys.
        VU = np.sum(np.where(peaks, peak_diffs, 0) + np.where(valleys, valley_diffs, 0))

        # Return the final VU value.
        return VU

    @cached_property
    def low_short_time_energy_ratio(self) -> float:
        """
        Compute the low short time energy of the audio signal.
        :return: array of low short time energies of each channel
        """

        # specify the number of following elements to include in each mean
        n = 1000 // self.frame_length_ms
        kernel = np.ones(n) / n
        arr_padded = np.concatenate((self.short_time_energy, np.zeros(n)))

        # compute the means using the convolve function
        arr_means = np.convolve(arr_padded, kernel, mode='valid')[
                    ::(len(self.short_time_energy) - n + 1)]
        print(
            f'n: {n}, arr_means: {arr_means.shape}, short_time_energy: {self.short_time_energy.shape}')
        # Compute low energy frames
        low_energy_frames = np.sign(
            0.5 * np.mean(arr_means) - self.short_time_energy
        )
        lster = np.mean(low_energy_frames + 1) / 2
        return lster

    @cache
    def energy_entropy(self, k: int = 10) -> float:
        """
        Compute the energy entropy of the audio signal.
        :param k: number of segments to split each frame into
        :return: array of energy entropies of each channel
        """
        # Split each frame into k segments
        frames_split = np.array_split(self.frames, k, axis=1)

        # Compute energy of each segment and normalize by total energy in a frame
        energies = np.array(
            [np.sum(segment ** 2) / np.sum(frame ** 2) for frame in frames_split for segment
             in frame])

        # Compute probability density function of energy values
        pdf = energies / np.sum(energies)
        # fix for "divide by zero encountered in log2"
        pdf[pdf == 0] = 10 ** (-100)

        # Compute energy entropy
        entropy = -np.sum(pdf * np.log2(pdf))

        return entropy

    @cached_property
    def zstd(self) -> float:
        """
        Compute the standard deviation of the zero crossing rate of the audio signal.
        :return: array of standard deviations of zero crossing rates of each channel
        """
        return self.zero_crossing_rate.std()

    @cached_property
    def hzcrr(self) -> float:
        """
        Compute the high zero crossing rate ratio of the audio signal.
        :return: array of high zero crossing rate ratios of each channel
        """
        return (
                np.mean(
                    np.sign(
                        self.zero_crossing_rate - 1.5 * np.mean(self.zero_crossing_rate) + 1
                    ),
                )
                / 2
        )

    @cached_property
    def sound_presence(self, threshold: float = 0.1):
        """
        Detect sound in the audio signal.
        :param threshold: threshold for detecting sound
        :return: array of sound detection values
        """
        classified_frames = np.zeros_like(self.short_time_energy)
        classified_frames[self.short_time_energy >= threshold] = 1
        return classified_frames

    @cache
    def get_audio_type(self, threshold: float = 0.45):
        """
        Detect audio type in the audio signal.
        :param threshold: threshold for detecting sound
        :return: array of sound detection values
        """
        return "Music" if self.low_short_time_energy_ratio < threshold else "Speech"


def split_samples_into_frames(samples: np.ndarray,
                              frame_size: int,
                              frame_overlap: int = 0) -> np.ndarray:
    """
    Split the samples array into frames of given size with given overlap.
    """
    frames = [
        samples[i: i + frame_size]
        for i in range(
            0, len(samples) - frame_size, frame_size - frame_overlap
        )
    ]
    return np.array(frames, dtype=np.int32)