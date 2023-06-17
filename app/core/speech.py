from functools import cache, cached_property

import numpy as np
from hmmlearn import hmm
from app.models.signal import Signal, split_samples_into_frames
from scipy.fftpack import dct


class HMMClassifier(object):
    def __init__(self, model_name='GMMHMM', n_components=4, n_mix=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.n_mix = n_mix
        self.models = []

    def fit(self, X, label):
        np.random.seed(42)
        X = self._extract_mfcc_features(X)
        model = hmm.GMMHMM(n_components=self.n_components,
                               n_mix=self.n_mix).fit(X)
        model.name = label
        self.models.append(model)

    def predict(self, X):
        np.random.seed(42)
        self._extract_mfcc_features(X)
        scores = [(m.score(X), m.name) for m in self.models]
        predicted_label, model_name = max(scores, key=lambda x: x[0])[1]
        return predicted_label, model_name

    def _extract_mfcc_features(self, signal: Signal):
        emphasized_sig = self._pre_emphasis(signal.samples)
        frames = split_samples_into_frames(emphasized_sig)
        frames = self._apply_window(frames)
        pow_frames = self._power_spectrum(frames)
        filter_banks = self._apply_filter_banks(pow_frames, signal.sample_rate)
        mfcc = self._compute_mfcc_coefficients(filter_banks)
        return mfcc

    @cache
    def _pre_emphasis(self, sig, pre_emphasis=0.97):
        emphasized_sig = np.append(sig[0], sig[1:] - pre_emphasis * sig[:-1])
        return emphasized_sig

    def _apply_window(self, frames):
        frames *= np.hamming(frames.shape[1])
        return frames

    def _power_spectrum(self, frames):
        mag_frames = np.absolute(np.fft.rfft(frames, n=512))
        pow_frames = ((1.0 / 512) * ((mag_frames) ** 2))
        return pow_frames

    def _apply_filter_banks(self, pow_frames, sample_rate, n_filters=26):
        n_frames, n_bins = pow_frames.shape

        mel_filterbank = self._mel_filterbank(n_filters, n_bins, sample_rate)

        filtered_frames = np.dot(pow_frames, mel_filterbank.T)
        filtered_frames = np.where(
            filtered_frames == 0, np.finfo(float).eps, filtered_frames)

        return filtered_frames
    
    @cached_property
    def _mel_filterbank(self, n_filters, n_bins, sample_rate, low_freq=0, high_freq=None):
        if high_freq is None:
            high_freq = sample_rate / 2

        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)

        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)

        bins = np.floor((n_bins + 1) * hz_points / sample_rate).astype(int)

        filterbank = np.zeros((n_filters, n_bins))
        for m in range(1, n_filters + 1):
            left = bins[m - 1]
            center = bins[m]
            right = bins[m + 1]

            # left slope
            filterbank[m - 1, :left] = 0
            filterbank[m - 1, left:center] = (
                np.arange(left, center) - bins[m - 1]) / (center - bins[m - 1])

            # right slope
            filterbank[m-1, right:n_bins] = (bins[m + 1] -
                                             np.arange(right, n_bins)) / (bins[m + 1] - center)
            filterbank[m-1, center:right] = 1
        return filterbank

    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def _compute_mfcc_coefficients(self, filter_banks):
        n_filters = filter_banks.shape[1]
        n_coefficients = n_filters - 1

        log_filter_banks = np.log(filter_banks)

        mfcc_coefficients  = dct(log_filter_banks, type=2, axis=1, norm='ortho')[
            :, 1: (n_coefficients + 1)]

        return mfcc_coefficients 
