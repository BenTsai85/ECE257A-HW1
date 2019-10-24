import numpy as np
import BPSK

class OFDM:
    def modulate(self, symbols):
        assert(symbols.shape[1] == 64)
        samples = np.fft.ifft(symbols, axis = 1)
        prefixed_samples = np.concatenate((samples[:, -16:], samples), axis = 1)
        return prefixed_samples.flatten()

    def demodulate(self, samples):
        samples = np.reshape(samples, (-1, 80))
        samples = samples[:, 16:]
        return np.fft.fft(samples, axis = 1)
