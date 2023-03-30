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
            
            # TODO - set appropriate frame size and overlap
            self.frames = self.__split_into_frames(1024, 0)

    @property
    def audio_length_sec(self) -> float:
        """
        > The function returns the length of the audio in seconds
        :return: The length of the audio in seconds.
        """
        return self.n_samples / self.sample_rate

    def __normalise_samples(self):
        """
        Normalize the samples to a target level in decibels.
        """
        max_amplitude = np.max(self.samples)
        target_level_db = -3
        target_amplitude = 10 ** (target_level_db / 20) * (2 ** (self.sample_width * 8 - 1) - 1)

        gain = target_amplitude / max_amplitude
        samples_normalised = np.floor(self.samples * gain)

        self.samples = samples_normalised

    
    def __split_into_frames(self, frame_size: int, overlap: int) -> np.ndarray:
        """
        Split the samples array into frames of given size with given overlap.
        :param frame_size: size of the frame in samples
        :param overlap: overlap between frames in samples
        :return: array of frames
        """
        frames = []
        for i in range(0, self.n_samples - frame_size, frame_size - overlap):
            frames.append(self.samples[i:i + frame_size])

        return np.array(frames, dtype=object)
    
    @property
    def volume(self) -> np.ndarray:
        """
        Compute the volume of the audio signal.
        :return: array of volumes of each channel
        """
        return np.sqrt(np.mean(self.frames ** 2, axis=1))

    @property
    def stereo_balance(self) -> np.ndarray:
        """
        Compute the stereo balance of the audio signal.
        :return: array of stereo balances of each channel
        """
        return self.frames[0] / self.frames[1]
    
    @property
    def short_time_energy(self) -> np.ndarray:
        """
        Compute the short time energy of the audio signal.
        :return: array of short time energies of each channel
        """
        return np.mean(self.frames ** 2, axis=1)

    @property
    def zero_crossing_rate(self) -> np.ndarray:
        """
        Compute the zero crossing rate of the audio signal.
        :return: array of zero crossing rates of each channel
        """
        return np.mean(np.abs(np.diff(np.sign(self.frames))), axis=1)
    
    # TODO - implement this
    @property
    def silence_rate(self) -> np.ndarray:
        """
        Compute the silence rate of the audio signal.
        :return: array of silence rates of each channel
        """
        return np.mean(np.abs(self.frames) < 0.01, axis=1)
    
#     # Miara ta wyliczana jest z głośności i ZCR. Jeżeli głośność (Volume) i ZCR dla ramki są poniżej
# # pewnego poziomu, ramka taka może zostać zaklasyfikowana jako cisza

# def get_silent_rate(zcr_frame: float, vol_frame: float, frame_length: int, threshold=0.0001):
#     """Computes the silent rate of audio frames.

#     Args:
#         frame (numpy.ndarray): Input audio frames.
#         threshold (float): The threshold below which a frame is considered silent.
#             Defaults to 0.0001.

#     Returns:
#         float: The silent rate of the input frames.
#     """
#     num_silent_frames = 0
#     for sample in frame:
#         if zcr_frame < threshold and vol_frame < threshold: #osobne thresholdy
#             num_silent_frames += 1
#     silent_rate = num_silent_frames / frame_length
#     return silent_rate

    @property
    def fundamental_frequency(self) -> np.ndarray:
        """
        Compute the fundamental frequency of the audio signal.
        :return: array of fundamental frequencies of each channel
        """
        # TODO - implement this
        return np.array([0, 0])
    
    @property
    def vstd(self) -> np.ndarray:
        """
        Compute the variance of the standard deviation of the audio signal.
        :return: array of variances of standard deviations of each channel
        """
        # TODO - implement this
        return np.var(np.std(self.frames, axis=1), axis=0)
    
    @property
    def volume_dynamic_range(self) -> np.ndarray:
        """
        Compute the volume dynamic range of the audio signal.
        :return: array of volume dynamic ranges of each channel
        """
        return 1 - np.min(self.frames, axis=1) / np.max(self.frames, axis=1)
    
    @property
    def volume_undulation(self) -> np.ndarray:
        """
        Compute the volume undulation of the audio signal.
        :return: array of volume undulations of each channel
        """
        # TODO - implement this
        return np.std(self.frames, axis=1)
    
    @property
    def low_short_time_energy(self) -> np.ndarray:
        """
        Compute the low short time energy of the audio signal.
        :return: array of low short time energies of each channel
        """
        # TODO - implement this
        return np.array([0, 0])