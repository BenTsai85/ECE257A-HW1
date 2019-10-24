import numpy as np
import math

class Distortion:
    def apply(self, data):
        data = self._apply_magnitude_distortion(data)
        data = self._apply_phase_shift(data)
        data = self._apply_frequency_offset(data)
        data = self._apply_channel_noise(data)
        return data

    def _apply_magnitude_distortion(self, data):
        return (10 ** (-5)) * data
    def _apply_phase_shift(self, data):
        return np.exp(-3j/4 * math.pi) * data
    def _apply_frequency_offset(self, data):
        return np.multiply(data, np.exp(-2j * math.pi * 0.00017 * np.arange(1, data.shape[0] + 1)))
    def _apply_channel_noise(self, data):
        return np.add(data, np.random.normal(loc = 0, scale = 10 ** (-7)))

    def recover_frequency_offset(self, data, ltf_start):
        ltf1 = data[ltf_start : ltf_start + 64]
        ltf2 = data[ltf_start + 64 : ltf_start + 128]
        freq_offset = np.sum(np.imag(np.divide(ltf1, ltf2))) / 2 / math.pi / 64 / 64
        print("Estimated Frequency Offset:\n{}".format(freq_offset))
        print("---------------------------------------------------------")
        return np.multiply(data, np.exp(2j * math.pi * freq_offset * np.arange(1, data.shape[0] + 1)))
    
    def get_channel_distortion(self, data, ltf_preamble, ltf_start):
        ltf1 = data[ltf_start + 32 : ltf_start + 96]
        ltf2 = data[ltf_start + 96: ltf_start + 160]
        ltf1 = np.fft.fft(ltf1)
        ltf2 = np.fft.fft(ltf2)
        return np.multiply(np.add(ltf1, ltf2) / 2, ltf_preamble)
