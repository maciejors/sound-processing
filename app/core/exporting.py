import json

from app.core.freq_analysis import FrequencyAnalyser
from app.models.signal import Signal


class Bundler:
    """Bundles selected functionalities and provides methods to export sound
    properties to a JSON format"""

    __slots__ = ['signal', '_incl_frame_level', '_incl_clip_level', '_freq']

    def __init__(self, signal: Signal, frame_level=True, clip_level=True, freq=True):
        self.signal = signal
        self._incl_frame_level = frame_level
        self._incl_clip_level = clip_level
        self._freq = FrequencyAnalyser(signal) if freq else None

    def export_json(self) -> str:
        """Dumps all sound features from bundled functionalities into one JSON string"""
        dict_features = {}
        if self._incl_frame_level:
            dict_features['frame-level'] = {
                "volume": list(self.signal.volume),
                "short_time_energy": list(self.signal.short_time_energy),
                "zero_crossing_rate": list(self.signal.zero_crossing_rate),
                "silence_rate": self.signal.get_silence_rate(),
                "fundamental_frequency": list(self.signal.fundamental_frequency),
            }
        if self._incl_clip_level:
            dict_features['clip-level'] = {
                "vstd": self.signal.vstd,
                "volume_dynamic_range": self.signal.volume_dynamic_range,
                "volume_undulation": self.signal.volume_undulation,
                "low_short_time_energy_ratio": self.signal.low_short_time_energy_ratio,
                "energy_entropy": self.signal.energy_entropy(),
                "zstd": self.signal.zstd,
                "hzcrr": self.signal.hzcrr,
            }
        if self._freq is not None:
            dict_features['freq-analysis'] = {
                'volume': list(self._freq.volume()),
                'frequency_cetroids': list(self._freq.frequency_cetroids()),
                'effective_bandwidth': list(self._freq.effective_bandwidth()),
                'ersb1': list(self._freq.ersb1()),
                'ersb2': list(self._freq.ersb2()),
                'ersb3': list(self._freq.ersb3()),
                'spectral_flatness': list(self._freq.spectral_flatness()),
                'spectral_crest_factor': list(self._freq.spectral_crest_factor()),
            }
        return json.dumps(dict_features)


